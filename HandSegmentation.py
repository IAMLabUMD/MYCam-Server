"""
hand segmentation codes

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import argparse
import csv
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jhong12/TOR-app-files/HandSegmentationFiles')
sys.path.insert(1, '/home/jhong12/TOR-app-files/HandSegmentationFiles/HandSegIncludes')

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import cv2
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from fractions import Fraction
from sklearn.metrics import average_precision_score

import TOR_utils

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
                    
from kitti_devkit import seg_utils as seg

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

red_color = [0, 0, 255, 127]
green_color = [0, 255, 0, 127]
blue_color = [255, 0, 0, 127]
gb_color = [255, 255, 0, 127]

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="localizer model")
parser.add_argument("--threshold", help="threshold for the model")
parser.add_argument("--width", help="image width")
parser.add_argument("--height", help="image height")
parser.add_argument("--input_file", help="file containing a list of images")
parser.add_argument("--input_image", help="input image")
parser.add_argument("--analysis", default=False, action='store_true',
                    help="Flag to perform analysis")
parser.add_argument("--csv", help="CSV file to analyze")

args = parser.parse_args()


class Segmentation:

    # constructor
    def __init__(self,
                 model='TOR_hand_all_fcn8_10k_1e-6_600x450',
                 logdir=None,
                 threshold=0.5,
                 image_width=None,
                 image_height=None,
                 debug=False):
        # flag for debugging mode
        self.debug = debug

        # set the log directory
        if logdir:
            if self.debug:
                logging.info("Using weights found in {}".format(logdir))
            self.logdir = logdir
        else:
            if 'TV_DIR_RUNS' in os.environ:
                runs_dir = os.path.join(os.environ['TV_DIR_RUNS'], 'KittiSeg')
            else:
                runs_dir = 'RUNS'
            self.logdir = os.path.join(runs_dir, model)

        # threshold to filter inference output
        self.threshold = float(threshold)

        # resizing parameters
        self.image_width = int(image_width) if image_width is not None else None
        self.image_height = int(image_height) if image_height is not None else None

        # Loading hyperparameters from logdir
        hypes = tv_utils.load_hypes_from_logdir(self.logdir, base_path='hypes')
        logging.info("Hypes loaded successfully.")

        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        modules = tv_utils.load_modules_from_logdir(self.logdir)
        logging.info("Modules loaded successfully. Starting to build tf graph.")

        # set to allocate memory on GPU as needed
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # Create tf graph and build module.
        with tf.Graph().as_default():
            # with tf.device('/cpu:0'):
            # Create placeholder for input
            self.input_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(self.input_pl, 0)

            # build Tensorflow graph using the model from logdir
            self.output_operation = core.build_inference_graph(hypes, modules, image=image)
            """
            self.input_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(self.input_pl, 0)

            # build Tensorflow graph using the model from logdir
            self.output_operation = core.build_inference_graph(hypes, modules,
                                                            image=image)
            """
            logging.info("Graph build successfully.")

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session(config=config)
            self.saver = tf.train.Saver()

            # Load weights from logdir
            core.load_weights(self.logdir, self.sess, self.saver)

            logging.info("Weights loaded successfully.")

    """
    Localize an object of interest in an image
    @input  :	 image (numpy array)
    @output :	 object image (numpy array)
    """

    def do(self, image):
        # read the input image file into numpy array
        shape = image.shape
        if self.debug:
            print("DEBUG: image shape = (", str(shape[0]), str(shape[1]), ")")

        # resize the input image if set
        if self.image_width is not None and self.image_height is not None and (
                self.image_width != shape[0] or self.image_height != shape[1]):
            image = cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_CUBIC)
        # start localizing an object of interest
        feed = {self.input_pl: image}
        # botfh 'softmax' and 'sigmoid' use 'output'
        pred = self.output_operation['output']

        logging.info("Running hand segmentation")
        output = self.sess.run([pred], feed_dict=feed)

        # reshape output from flat vector to 2D Image
        shape = image.shape
        # print("max prob: ", max(output[0]))
        output_image = output[0][:, 1]
        output_image = output_image.reshape(shape[0], shape[1])

        # if self.debug:
        #	overlaid_img = image.copy()
        #	cv2.imwrite("segRawOutput.png", output_image)
        #	green_color = [0, 255, 0, 127]
        #	pred = output_image > self.threshold
        #	seg_image = tv_utils.fast_overlay(overlaid_img, pred, green_color)
        #	rb_image = seg.make_overlay(image, output_image)
        #	cv2.imwrite("segRBOutput.png", rb_image)
        #	cv2.imwrite("segmentedOutput.png", seg_image)
        #	logging.info("saved localizer's intermediate results")

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        logging.info("Given threshold value: %f" % self.threshold)
        logging.info("Max threshold value: %f" % np.max(output_image))

        pred = output_image > self.threshold

        # getting x y index for drawing a bouding box
        # max_index = np.unravel_index(np.argmax(output_image, axis=None),
        #							  output_image.shape)

        return image, pred

    """
    Calculate Intersection Over Union (IOU) score
    @input  : gt (numpy array), pred (numpy array)
    @output : iou (float)
    """

    def get_iou(self, gt, pred):
        intersection = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        return np.sum(intersection) / np.sum(union)

    def eval(self, gt, pred):
        """ Evaluate the hand segmentation model

        Args:
          gt:		  (numpy array)
          left_pred:  (numpy array)
          right_pred: (numpy array)

        Returns:
          iou (float)
        """
        shape = gt.shape
        # resize the gt image if set
        if self.image_width is not None and self.image_height is not None and (
                self.image_width != shape[0] or self.image_height != shape[1]):
            gt = cv2.resize(gt, (self.image_width, self.image_height), cv2.INTER_CUBIC)

        assert (gt.shape[0] == pred.shape[0] and gt.shape[1] == pred.shape[1])

        left_hand = [100, 100, 100]
        right_hand = [150, 150, 150]
        both_hand = [200, 200, 200]
        hand_color1 = [222, 222, 222],
        hand_color2 = [255, 255, 255],
        # ground-truth of left hand and right hand
        left_gt = np.all(gt == left_hand, axis=2)
        right_gt = np.all(gt == right_hand, axis=2)
        both_gt = np.all(gt == both_hand, axis=2)
        gt1 = np.all(gt == hand_color1, axis=2)
        gt2 = np.all(gt == hand_color2, axis=2)

        hand_gt = left_gt | right_gt | both_gt | gt1 | gt2
        iou = -1.
        y_true = y_est25 = y_est50 = y_est75 = 0
        if np.any(hand_gt):
            # set y_true to 1 as it has ground-truth annotation
            y_true = 1
            # calculate IOU only if there is ground-truth
            iou = self.get_iou(hand_gt, pred)
            if iou > 0.75:
                y_est75 = iou
            if iou > 0.5:
                y_est50 = iou
            if iou > 0.25:
                y_est25 = iou
        else:
            if np.any(pred):
                y_est25 = y_est50 = y_est75 = 1

        return iou, y_true, y_est25, y_est50, y_est75


"""
main function
"""


def main():
#     model = 'TOR_hand_all+TEgO_fcn8_10k_16_1e-5_450x450'
    model ='TOR_hand_all+TOR_feedback_fcn8_10k_16_1e-5_450x450'
    threshold = 0.5
    width = 450
    height = 450

    # initialize Localizer and Classifier
    debug = True
    segmentation = Segmentation(model=model,
                                threshold=threshold,
                                image_width=width,
                                image_height=height,
                                debug=debug)

    image = cv2.imread('/Users/jonggihong/Downloads/tmp_2.jpg', cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        new_shape = (int(width), int(height))
        image = cv2.resize(image, new_shape, cv2.INTER_CUBIC)

    # localize an object from the input image
    image, pred = segmentation.do(image)
    hand_area = np.sum(pred)