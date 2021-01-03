'''
    A class to generate descriptor of an image or a set of imagess

    Author: Jonggi Hong
    Date: 01/03/2020
'''
import numpy as np
import cv2
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/data/kyungjun/TOR/hand-guided/')

class DescriptorGenerator:
    def __init__(self):
        pass

    def getImageDescriptor(self, img_path):
        hand_area = self.getHandSegment(img_path)
        print(img_path, hand_area)
        pass

    def getHandSegment(self, img_path):
        model = 'TOR_hand_all+TEgO_fcn8_10k_16_1e-5_450x450'
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

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if width is not None and height is not None:
            new_shape = (int(width), int(height))
            image = cv2.resize(image, new_shape, cv2.INTER_CUBIC)

        # localize an object from the input image
        image, pred = segmentation.do(image)
        hand_area = np.sum(pred)
        return hand_area

    def getSetDescriptor(self, set_path):
        pass


if __name__ == '__main__':
    dg = DescriptorGenerator()
    dg.getImageDescriptor('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg')
