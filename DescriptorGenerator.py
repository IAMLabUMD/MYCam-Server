'''
    A class to generate descriptor of an image or a set of imagess

    Author: Jonggi Hong
    Date: 01/03/2020
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
import sys
from HandSegmentation import Segmentation
from ObjectDetector import ObjectDetector
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import statistics

    
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
	
	'''
		generate the image descriptor
	'''
	def getImageDescriptor(self, img_path):
		th_hand = 1343 # 0.0034542181
		hand = False
		hand_area = self.getHandArea(img_path)
		if hand_area > th_hand:
			hand = True
		
		blurry = False
# 		th_blur = 29.2
		th_blur = 3
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
		
		desc_obj = {}
		desc_obj['hand_area'] = hand_area
		desc_obj['blurriness'] = blurriness
		desc_obj['boxes'] = boxes
		desc_obj['img_width'] = img_width
		desc_obj['img_height'] = img_height
		
		return hand, blurry, cropped, small, desc_obj
# 		return hand_area, blurriness, 
	
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

	'''
		generate the set descriptor
	'''
	def getSetDescriptor(self, arinfo_path):
		arinfo = self.loadARInfo(arinfo_path)
		cam_pos_sd, cam_ori_sd = self.computeBackgroundVariation(arinfo)
		side_num = self.computeSideVariation(arinfo)
		dist_sd = self.computeDistanceVariation(arinfo)
		hand, blurry, cropped, small = self.countImgDescriptors(arinfo)
		
# 		bg_var = True if cam_pos_sd > 0.1 or cam_ori_sd > 0.1 else False
# 		side_var = True if side_num > 1 else False
# 		dist_var = True if dist_sd > 0.1 else False
# 		
# 		return bg_var, side_var, dist_var, hand, blurry, cropped, small

# 		bg_var = min(max(cam_pos_sd/0.15, cam_ori_sd / 0.15), 1.0) * 100
# 		side_var = min(side_num/1.5, 1.0) * 100
# 		dist_var = min(dist_sd/0.15, 1.0) * 100
		bg_var = max(cam_pos_sd, cam_ori_sd)
		side_var = side_num
		dist_var = dist_sd
		
		return bg_var, side_var, dist_var, hand, blurry, cropped, small

	'''
		compute the background variation using the AR information.
		The background variation is the variation of camera orientation and position
	'''
	def computeBackgroundVariation(self, arinfo):
		pos_diff = []
		orientation_diff = []
		
		for img_id1, img_info1 in arinfo.items():
			for img_id2, img_info2 in arinfo.items():
				if img_id1 < img_id2:
					cp1 = img_info1['camera_position']
					cp2 = img_info2['camera_position']
					pd = euclidean_distances([cp1], [cp2])[0][0]
					pos_diff.append(pd)
					
					co1 = img_info1['camera_orientation']
					co2 = img_info2['camera_orientation']
					od = 1-cosine_similarity([co1], [co2])[0][0]
					orientation_diff.append(od)
		
# 		print(statistics.stdev(pos_diff),statistics.stdev(orientation_diff))
		return statistics.stdev(pos_diff),statistics.stdev(orientation_diff)
	
	def computeSideVariation(self, arinfo):
		sides = []
		for img_id, img_info in arinfo.items():
			ar_side = img_info['ar_side']
			if not ar_side in sides:
				sides.append(ar_side)
# 		print(sides)
		return len(sides) - 1
	
	def computeDistanceVariation(self, arinfo):
		dist_diff = []
		
		for img_id1, img_info1 in arinfo.items():
			for img_id2, img_info2 in arinfo.items():
				dist1 = euclidean_distances([img_info1['obj_cam_position']], [(0, 0, 0)])[0][0]
				dist2 = euclidean_distances([img_info2['obj_cam_position']], [(0, 0, 0)])[0][0]
				dd = abs(dist1-dist2)
				
				if not dd in dist_diff:
					dist_diff.append(dd)
		
		if len(dist_diff) > 1:
# 			print(statistics.stdev(dist_diff))
			return statistics.stdev(dist_diff)
		else:
# 			print('0')
			return 0
	
	def countImgDescriptors(self, arinfo):
		hand, blurry, cropped, small = 0, 0, 0, 0
		for img_id, img_info in arinfo.items():
			if img_info['hand'] == 'True':
				hand += 1
			if img_info['blurry'] == 'True':
				blurry += 1
			if img_info['cropped'] == 'True':
				cropped += 1
			if img_info['small'] == 'True':
				small += 1
		
# 		print(hand, blurry, cropped, small)
		return hand, blurry, cropped, small
	
	'''
		load AR information from a text file.
		format: 
		var desc_info = "\(count)#\(self.ar_side)#" + //1
						"\(self.camera_position.0)#\(self.camera_position.1)#\(self.camera_position.2)," + // 4
						"\(self.camera_orientation.0)#\(self.camera_orientation.1)#\(self.camera_orientation.2)" + // 7
						"\(self.obj_position.0)#\(self.obj_position.1)#\(self.obj_position.2)," + // 10
						"\(self.obj_orientation.0)#\(self.obj_orientation.1)#\(self.obj_orientation.2)" + // 13
						"\(self.obj_cam_position.0)#\(self.obj_cam_position.1)#\(self.obj_cam_position.2)" + // 16
						"\(cam_mat)#\(obj_mat)#\(cam_obj_mat)" // 19
	'''
	def loadARInfo(self, arinfo_path):
		arinfo = {}
		
		f = open(arinfo_path, "r")
		for line in f:
			words = line.split('#')
			for i, w in enumerate(words):
				print(i, w)
			print()
			img_id = int(words[0])
			arinfo[img_id] = {}
			arinfo[img_id]['ar_side'] = words[1]
			arinfo[img_id]['camera_position'] = (float(words[2]), float(words[3]), float(words[4]))
			arinfo[img_id]['camera_orientation'] = (float(words[5]), float(words[6]), float(words[7]))
			arinfo[img_id]['obj_cam_position'] = (float(words[14]), float(words[15]), float(words[16]))
			arinfo[img_id]['hand'] = words[20]
			arinfo[img_id]['blurry'] = words[21]
			arinfo[img_id]['cropped'] = words[22]
			arinfo[img_id]['small'] = words[23]
		
		return arinfo
			
		
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
	dg.initialize()
# 	dg.getBlurriness('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg')

# 	print('Omega3 - with variation')
# 	train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Spice/Omega3'
# 	arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Omega3/desc_info.txt'
# 	print(dg.getSetDescriptor(train_img_dir, arinfo_path))
# 
# 	print()
# 	print('Knife - no variation')
# 	train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Spice/Knife'
# 	arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Knife/desc_info.txt'
# 	print(dg.getSetDescriptor(train_img_dir, arinfo_path))

	train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/B2803393-73CE-4F25-B9F1-410D2A37D0DE/Spice/Knife'
	arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/B2803393-73CE-4F25-B9F1-410D2A37D0DE/Knife/desc_info.txt'
	print(dg.getSetDescriptor(train_img_dir, arinfo_path))
	
	train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/74DBAC2E-79F5-4C39-B281-7719602D54BC/Spice/Mouse'
	arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/74DBAC2E-79F5-4C39-B281-7719602D54BC/Mouse/desc_info.txt'
	print(dg.getSetDescriptor(train_img_dir, arinfo_path))
