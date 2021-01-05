'''
    A class to generate descriptor of an image or a set of imagess

    Author: Jonggi Hong
    Date: 01/03/2020
'''
import numpy as np
import cv2
import sys
from HandSegmentation import Segmentation
from ObjectDetector import ObjectDetector


class DescriptorGenerator:
	def __init__(self):
		pass

	def initialize(self):
		model = 'TOR_hand_all+TEgO_fcn8_10k_16_1e-5_450x450'
		threshold = 0.5
		self.input_width = 450
		self.input_height = 450

		# initialize Localizer and Classifier
		debug = False
		self.segmentation = Segmentation(model=model,
									threshold=threshold,
									image_width=self.input_width,
									image_height=self.input_height,
									debug=debug)
		self.object_detector = ObjectDetector()
	
	def getImageDescriptor(self, img_path):
		th_hand = 1343 # 0.0034542181
		hand = False
		hand_area = self.getHandArea(img_path)
		if hand_area > th_hand:
			hand = True
		
		blurry = False
		th_blur = 29.2
		blurriness = self.getBlurriness(img_path)
		if blurriness < th_blur:
			blurry = True
		
		cropped = False
		small = False
		boxes, img_width, img_height = self.object_detector.detect(img_path)
		if len(boxes) == 1:
			if self.isCropped(boxes[0], img_width, img_height):
				cropped = True
			if self.isSmall(boxes[0], img_width, img_height):
				small = True
		
		return hand, blurry, cropped, small
		
	
	def isSmall(self, box, img_width, img_height):
		box_w = box['xmax']-box['xmin']
		box_y = box['ymax']-box['ymin']
		if box_w*box_y/(img_width+1)*(img_height+1) < 0.125:
			return True
		return False
	
	def isCropped(self, box, img_width, img_height):
		if box['xmin'] < 0.02 * img_width or box['ymin'] < 0.02 * img_height or box['xmax'] > 0.98 * img_width or box['ymax'] > 0.98 * img_height:
			return True
		return False

	def getHandArea(self, img_path):
		image = cv2.imread(img_path, cv2.IMREAD_COLOR)
		if self.input_width is not None and self.input_height is not None:
			new_shape = (int(self.input_width), int(self.input_height))
			image = cv2.resize(image, new_shape, cv2.INTER_CUBIC)

		# localize an object from the input image
		image, pred = self.segmentation.do(image)
		hand_area = np.sum(pred)
		return hand_area

	def getSetDescriptor(self, set_path):
		pass

	# higher value means more blurriness
	def getBlurriness(self, img_path):
		image = cv2.imread(img_path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# compute the Laplacian of the image and then return the focus
		# measure, which is simply the variance of the Laplacian
		blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
		print('blur:', blurriness)
		return blurriness


if __name__ == '__main__':
    dg = DescriptorGenerator()
    dg.getBlurriness('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg')
