# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""------------------- Modified for CHI2017 Experiments ------------------------
Simple transfer learning with an Inception v3 architecture model.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:

bazel build third_party/tensorflow/examples/image_retraining:retrain && \
bazel-bin/third_party/tensorflow/examples/image_retraining/retrain \
--image_dir ~/flower_photos

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from collections import OrderedDict, defaultdict
import glob
import hashlib
import os.path
import random
import re
import sys
import csv
import json
import tarfile

import numpy as np
from six.moves import urllib
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from tensorflow.python.client import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

# calculate recall and precision
from sklearn.metrics import precision_score, recall_score, f1_score

# file lock on writing csv files
import fcntl

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

FLAGS = tf.app.flags.FLAGS

# Input and output file flags.
tf.app.flags.DEFINE_string('image_dir', '/home/jhong12/ASSETS2019/p01/original',
                           """Path to folders of labeled images.""")
tf.app.flags.DEFINE_string('output_graph', '/home/jhong12/GORTestModels/tmp/model.pb',
                           """Where to save the trained graph.""")
tf.app.flags.DEFINE_string('output_labels', '/home/jhong12/GORTestModels/tmp/labels.txt',
                           """Where to save the trained graph's labels.""")

# Details of the training configuration.
tf.app.flags.DEFINE_integer('how_many_training_steps', 1000,
                            """How many training steps to run before ending.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """How large a learning rate to use when training.""")
tf.app.flags.DEFINE_integer(
    'testing_percentage', 10,
    """What percentage of images to use as a test set.""")
tf.app.flags.DEFINE_integer(
    'validation_percentage', 10,
    """What percentage of images to use as a validation set.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 50,
                            """How often to evaluate the training results.""")
tf.app.flags.DEFINE_integer('train_batch_size', 100,
                            """How many images to train on at a time.""")
tf.app.flags.DEFINE_integer('test_batch_size', 500,
                            """How many images to test on at a time. This"""
                            """ test set is only used infrequently to verify"""
                            """ the overall accuracy of the model.""")
tf.app.flags.DEFINE_integer(
    'validation_batch_size', 100,
    """How many images to use in an evaluation batch. This validation set is"""
    """ used much more often than the test set, and is an early indicator of"""
    """ how accurate the model is during training.""")

# File-system cache locations.
tf.app.flags.DEFINE_string('model_dir', '/home/jhong12/TOR-app-files/base_model',
                           """Path to classify_image_graph_def.pb, """
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('bottleneck_dir', '/home/jhong12/GORTestModels/tmp/bottleneck',
    """Path to cache bottleneck layer values as files.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")

# Controls the distortions used during training.
tf.app.flags.DEFINE_boolean(
    'flip_left_right', False,
    """Whether to randomly flip half of the training images horizontally.""")
tf.app.flags.DEFINE_integer(
    'random_crop', 0,
    """A percentage determining how much of a margin to randomly crop off the"""
    """ training images.""")
tf.app.flags.DEFINE_integer(
    'random_scale', 0,
    """A percentage determining how much to randomly scale up the size of the"""
    """ training images by.""")
tf.app.flags.DEFINE_integer(
    'random_brightness', 0,
    """A percentage determining how much to randomly multiply the training"""
    """ image input pixels up or down by.""")

# ASSETS2019 Experiment parameters
tf.app.flags.DEFINE_integer(
    'k_shot', 20,
    """Number of examples to be used for training.""")
tf.app.flags.DEFINE_integer(
    'attempt_number', 1,
    """A number denoting the attempt and used to get different random k.""")
tf.app.flags.DEFINE_string(
    'output_file', '/home/jhong12/GORTestModels/tmp/experiment_results.csv',
    """Where to write the results of the experiment.""")
tf.app.flags.DEFINE_integer(
    'gpu_to_use', 0,
    """GPU number to use""")

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
RANDOM_SEED_GRAPH = 37
RANDOM_SEED = 12
OUTPUT_HEADER = ["pid",
                 "input",
                 "start_time",
                 "start_training",
                 "end_time",
                 "image_dir",
                 "k_shot",
                 "attempt_number",
                 "random_crop",
                 "random_scale",
                 "random_brightness",
                 "flip_left_right",
                 "how_many_training_steps",
                 "learning_rate",
                 "train_batch_size",
                 "test_batch_size", 
                 "validation_batch_size",
                 "train_accuracy",
                 "cross_entropy",
                 "validation_accuracy",
                 "test1_accuracy",
                 "test1_f1_per_object",
                 "test1_recall_per_object",
                 "test1_precision_per_object",
                 "test2_accuracy",
                 "test2_f1_per_object",
                 "test2_recall_per_object",
                 "test2_precision_per_object",
                 "labels"] 

def create_image_lists(image_dir, k, attempt):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.
    k: Number of examples to be used for training.
    attempt: Attempt number. To make sure that that the random chooses different numbers in each attempt.
    

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = defaultdict(dict)
  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    training_file_list = []
    testing_file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print('current dir name', dir_name)

    if dir_name == 'train':
      phase = dir_name
      continue
    if dir_name == 'test1':
      phase = dir_name
      continue
    if dir_name == "test2":
      phase = dir_name

    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    if phase == "train":
      print("Looking for images in '" + 'train/'+dir_name + "'")
      for extension in extensions:
        file_glob = os.path.join(image_dir, 'train/'+dir_name, '*.' + extension)
        training_file_list.extend(glob.glob(file_glob))
      if not training_file_list:
        print('No files found')
        continue
      if len(training_file_list) < 30:
        print(len(training_file_list))
        print('WARNING: Folder has less than 30 images, which may cause issues.')
      training_validation_images = [os.path.basename(file_name) for file_name in training_file_list]
      # shuffle the images and choose top k for training
      # to make sure that the second attempt we don't choose the same images we introduce
      # the attempt number. 
      random.seed(RANDOM_SEED + attempt)
      random.shuffle(training_validation_images)
      result[label_name]['dir'] = dir_name
      result[label_name]['train'] = training_validation_images[:k]
      result[label_name]['validation'] = training_validation_images[k:]
      """
      result[label_name] = {
        'dir': dir_name,
        'training': training_validation_images[:k],
        'validation': training_validation_images[k:], 
        'testing': []}
      """
    elif phase == "test1":
      print("Looking for images in '" + 'test1/' + dir_name + "'")
      for extension in extensions:
        file_glob = os.path.join(image_dir, 'test1/' + dir_name, '*.' + extension)
        testing_file_list.extend(glob.glob(file_glob))
      if not testing_file_list:
        print('No files found')
        continue
      if len(testing_file_list) < 5:
        print(len(testing_file_list))
        print('WARNING: Folder has less than 5 images, which may cause issues.')  
      testing_images = [os.path.basename(file_name) for file_name in testing_file_list] 
      result[label_name]['test1'] = testing_images
      # result[label_name] =['testing'] = testing_images
    else:
      print("Looking for images in '" + 'test2/' + dir_name + "'")
      for extension in extensions:
        file_glob = os.path.join(image_dir, 'test2/' + dir_name, '*.' + extension)
        testing_file_list.extend(glob.glob(file_glob))
      if not testing_file_list:
        print('No files found')
        continue
      if len(testing_file_list) < 5:
        print(len(testing_file_list))
        print('WARNING: Folder has less than 5 images, which may cause issues.')  
      testing_images = [os.path.basename(file_name) for file_name in testing_file_list] 
      result[label_name]['test2'] = testing_images

  # return dictionary ordered by key  
  return OrderedDict(sorted(result.items(), key=lambda t: t[0]))

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Category has no images - %s.', category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  if 'train' in category or 'validation' in category:
    sub_dir = 'train/' + sub_dir
  elif 'test1' in category:
    sub_dir = 'test1/' + sub_dir
  else:
    sub_dir = 'test2/' + sub_dir
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.
  """ 
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def create_inception_graph(attempt_number):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
      # Set seed for a deterministic behavior at the graph level
      tf.set_random_seed(RANDOM_SEED_GRAPH + attempt_number)
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: Numpy array of image data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract():
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  if 'train' in category or 'validation' in category:
    sub_dir = 'train/' + sub_dir
  elif 'test1' in category:
    sub_dir = 'test1/' + sub_dir
  else:
    sub_dir = 'test2/' + sub_dir
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)

  if not os.path.exists(bottleneck_path):
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                jpeg_data_tensor,
                                                bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
      bottleneck_file.write(bottleneck_string)

  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['train', 'test1', 'test2', 'validation']:
      try:
        category_list = label_lists[category]
        for index, unused_base_name in enumerate(category_list):
          get_or_create_bottleneck(sess, image_lists, label_name, index,
                                   image_dir, category, bottleneck_dir,
                                   jpeg_data_tensor, bottleneck_tensor)
          how_many_bottlenecks += 1
          if how_many_bottlenecks % 100 == 0:
            print(str(how_many_bottlenecks) + ' bottleneck files created.')
      except KeyError:
        print("No files, skipping")
        continue


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The number of bottleneck values to return.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    # print(label_name)
    image_index = random.randrange(65536)
    bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                          image_index, image_dir, category,
                                          bottleneck_dir, jpeg_data_tensor,
                                          bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(65536)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_input_tensor,
                                         bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """

  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
  precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                  MODEL_INPUT_DEPTH])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """

  bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
      name='BottleneckInputPlaceholder')
  layer_weights = tf.Variable(
      tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
    name='final_weights')
  layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
  logits = tf.matmul(bottleneck_input, layer_weights,
                     name='final_matmul') + layer_biases
  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  ground_truth_input = tf.placeholder(tf.float32,
                                      [None, class_count],
                                      name='GroundTruthInput')
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=ground_truth_input)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy_mean)
  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor, layer_weights, layer_biases)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
  """
  correct_prediction = tf.equal(
      tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  return evaluation_step

def add_correct_prediction(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
  """
  correct_prediction = tf.equal(
      tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
  return correct_prediction


def add_evaluation_per_object(result_tensor, ground_truth_tensor, num_classes):
  """Insert the operations to calculate the precision per object

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data into.

  Returns:
    a list of precision_per_object
  """
  precisions = [0.] * num_classes
  recalls = [0.] * num_classes
  y_true = tf.argmax(ground_truth_tensor, 1)
  for i in range(num_classes):
    precision, _ = tf.metrics.precision_at_k(
        labels=y_true,
        predictions=result_tensor,
        k=1,
        class_id=i)
    recall, _ = tf.metrics.recall_at_k(
        labels=y_true,
        predictions=result_tensor,
        k=1,
        class_id=i)
    precisions[i] = precision
    recalls[i] = recall

  return precisions, recalls


def evaluation_per_object(predictions, labels):
  """Evaluate the performance (recall, precision) per class

  Args:
    predictions:
    labels:
    num_classes:

  Returns:
    recalls, precisions
  """
  y_true = np.argmax(labels, 1)
  y_pred = np.argmax(predictions, 1)
  f1s = f1_score(y_true, y_pred, average=None)
  recalls = recall_score(y_true, y_pred, average=None)
  precisions = precision_score(y_true, y_pred, average=None)

  return f1s, recalls, precisions


def get_pid_and_input_type(image_dir):
  """ Get pid and input_type from the image directory

  Args:
    image_dir: the input directory (e.g., study/pid/original/)

  Return:
    pid and input_type
  """
  dirs = image_dir.split('/')
  is_last_empty = dirs[-1] == ''
  if is_last_empty:
    pid, input_type = dirs[-2], dirs[-3]
  else:
    pid, input_type = dirs[-2], dirs[-1]

  return pid, input_type


def main(_):
  # setting GPU to use
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  if FLAGS.gpu_to_use >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu_to_use)
  else:
    os.environ["CUDA_VISIBLE_DEVICES"]=""

  start_time = datetime.now()
  # Set up the pre-trained graph.
  maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph(FLAGS.attempt_number))

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.k_shot, 
                                   FLAGS.attempt_number)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + FLAGS.image_dir +
          ' - multiple classes are needed for classification.')
    return -1

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)
  sess = tf.Session()

  if do_distort_images:
    # We will be applying distortions, so setup the operations we'll need.
    distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
  else:
    # We'll make sure we've calculated the 'bottleneck' image summaries and
    # cached them on disk.
    cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor)

  # Add the new layer that we'll be training.
  (train_step, cross_entropy, bottleneck_input, ground_truth_input,
   final_tensor, layer_weights, layer_biases) = add_final_training_ops(len(image_lists.keys()),
                                          FLAGS.final_tensor_name,
                                          bottleneck_tensor)

  # Create the operations we need to evaluate the accuracy of our new layer.
  evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)
  correct_prediction_step = add_correct_prediction(final_tensor, ground_truth_input)
  (precisions_step, recalls_step) = add_evaluation_per_object(final_tensor,
                                                              ground_truth_input,
                                                              class_count)
  # Set up all our weights to their initial default values.
  init = tf.global_variables_initializer()
  sess.run(init)
  sess.run(tf.local_variables_initializer())
  
  start_training = datetime.now()
  # Run the training for as many cycles as requested on the command line.
  for i in range(FLAGS.how_many_training_steps):
    # Get a catch of input bottleneck values, either calculated fresh every time
    # with distortions applied, or from the cache stored on disk.
    if do_distort_images:
      train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'train',
          FLAGS.image_dir, distorted_jpeg_data_tensor,
          distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
    else:
      train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'train',
          FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
          bottleneck_tensor)
    # Feed the bottlenecks and ground truth into the graph, and run a training
    # step.
    sess.run(train_step,
             feed_dict={bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth})
    # Every so often, print out how well the graph is training.
    is_last_step = (i + 1 == FLAGS.how_many_training_steps)
    if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
      train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                      train_accuracy * 100))
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                 cross_entropy_value))
      if FLAGS.validation_percentage > 0:
        validation_bottlenecks, validation_ground_truth = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                bottleneck_tensor))
        validation_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        print('%s: Step %d: Validation accuracy = %.1f%%' %
            (datetime.now(), i, validation_accuracy * 100))
  
  end_time = datetime.now()
  # We've completed all our training, so run a final test evaluation on
  # some new images we haven't used before.
  pid, input_type = get_pid_and_input_type(FLAGS.image_dir)

  test1_bottlenecks, test1_ground_truth = get_random_cached_bottlenecks(
      sess, image_lists, FLAGS.test_batch_size, 'test1',
      FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
      bottleneck_tensor)
  """
  test1_accuracy, test1_precisions, test1_recalls = sess.run(
      [evaluation_step, precisions_step, recalls_step],
      feed_dict={bottleneck_input: test1_bottlenecks,
                ground_truth_input: test1_ground_truth})
  """
  test1_prediction, test1_accuracy = sess.run(
      [final_tensor, evaluation_step],
      feed_dict={bottleneck_input: test1_bottlenecks,
                ground_truth_input: test1_ground_truth})

  test1_f1s, test1_recalls, test1_precisions = evaluation_per_object(
        test1_prediction, test1_ground_truth)
  
  # print(test1_f1s[0])
  # print(test1_recalls)
  # print(test1_precisions)
  # precision per object
  """
  test1_correct_prediction = sess.run(
      correct_prediction_step,
      feed_dict={bottleneck_input: test1_bottlenecks,
                ground_truth_input: test1_ground_truth})
  test1_prediction = sess.run(
      final_tensor,
      feed_dict={bottleneck_input: test1_bottlenecks,
                ground_truth_input: test1_ground_truth})
  """
  if pid != "p03":
    test2_bottlenecks, test2_ground_truth = get_random_cached_bottlenecks(
        sess, image_lists, FLAGS.test_batch_size, 'test2',
        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
        bottleneck_tensor)
    test2_prediction, test2_accuracy = sess.run(
        [final_tensor, evaluation_step],
        feed_dict={bottleneck_input: test2_bottlenecks,
                  ground_truth_input: test2_ground_truth})
    
    test2_f1s, test2_recalls, test2_precisions = evaluation_per_object(
          test2_prediction, test2_ground_truth)
  else:
    test2_accuracy = 0.
    test2_f1s = test2_recalls = test2_precisions = []

  # print(test2_f1s[0])
  # print(test2_recalls)
  # print(test2_precisions)
  """
  test2_correct_prediction = sess.run(
      correct_prediction_step,
      feed_dict={bottleneck_input: test2_bottlenecks,
                ground_truth_input: test2_ground_truth})
  test2_prediction = sess.run(
      final_tensor,
      feed_dict={bottleneck_input: test2_bottlenecks,
                ground_truth_input: test2_ground_truth})
  """
  
  #pred_values = sess.run(final_tensor)
  # print(test_bottlenecks)
  # print(test_ground_truth)
  # print(test_prediction)
  print('%s, %s, test1 accuracy = %.1f%%' % (pid, input_type, test1_accuracy * 100))
  print('%s, %s, test2 accuracy = %.1f%%' % (pid, input_type, test2_accuracy * 100))

  label_order = list(image_lists.keys())
  running_info = {'pid': pid,
                  'input': input_type,
                  'start_time': start_time, 
                  'start_training': start_training,
                  'end_time': end_time, 
                  'image_dir': FLAGS.image_dir,
                  'k_shot': FLAGS.k_shot, 
                  'attempt_number': FLAGS.attempt_number,
                  'random_crop': FLAGS.random_crop, 
                  'random_scale': FLAGS.random_scale, 
                  'random_brightness': FLAGS.random_brightness,
                  'flip_left_right': FLAGS.flip_left_right, 
                  'how_many_training_steps': FLAGS.how_many_training_steps,
                  'learning_rate': FLAGS.learning_rate,
                  'train_batch_size': FLAGS.train_batch_size,
                  'test_batch_size': FLAGS.test_batch_size, 
                  'validation_batch_size': FLAGS.validation_batch_size,
                  'train_accuracy': train_accuracy * 100,
                  'cross_entropy': cross_entropy_value,
                  'validation_accuracy': validation_accuracy*100,
                  'test1_accuracy': test1_accuracy * 100,
                  'test1_f1_per_object': test1_f1s,
                  'test1_recall_per_object': test1_recalls,
                  'test1_precision_per_object': test1_precisions,
                  'test2_accuracy': test2_accuracy * 100,
                  'test2_f1_per_object': test2_f1s,
                  'test2_recall_per_object': test2_recalls,
                  'test2_precision_per_object': test2_precisions,
                  'labels': label_order}
  
  if not os.path.exists(FLAGS.output_file):
    with open(FLAGS.output_file,'w') as f:
      fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
      # use csv.writer
      writer = csv.DictWriter(f, delimiter=';', fieldnames=OUTPUT_HEADER)
      writer.writeheader()
      writer.writerow(running_info)
      """
      f.write(','.join(OUTPUT_HEADER)+'\n')
      f.write(running_info)
      """
      fcntl.flock(f, fcntl.LOCK_UN)
  else:
    with open(FLAGS.output_file,'a') as f:
      fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
      # use csv.writer
      writer = csv.DictWriter(f, delimiter=';', fieldnames=OUTPUT_HEADER)
      writer.writerow(running_info)
      # f.write(running_info)
      fcntl.flock(f, fcntl.LOCK_UN)

  # Write out the trained graph and labels with the weights stored as constants.
  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph.as_graph_def(), [FLAGS.final_tensor_name])
  with tf.gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
    f.write('\n'.join(image_lists.keys()) + '\n')
  
  # Creates a saver.
  # saver0 = tf.train.Saver(var_list=[layer_weights, layer_biases])
  # saver0 = tf.train.Saver()
  # saver0.save(sess, '/tmp/imagenet/saver0.ckpt')
  # Generates MetaGraphDef.
  # saver0.export_meta_graph('/tmp/imagenet/my-model.meta', as_text=True)

  # Export the model to /tmp/my-model.meta.
  # meta_graph_def = tf.train.export_meta_graph(filename='/tmp/imagenet/my-model.meta')


if __name__ == '__main__':
  tf.app.run()