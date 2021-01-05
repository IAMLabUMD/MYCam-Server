'''
    Ab object detector class

    Author: Jonggi Hong
    Date: 01/04/2021
'''

import os
import sys

od_path = '/home/jhong12/TOR-app-files/ObjectDetectionFiles/'
src_path = od_path + '2_Training/src'
utils_path = od_path + 'Utils'

sys.path.append(src_path)
sys.path.append(utils_path)

print(sys.path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random


class ObjectDetector:
	def __init__(self):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		# Set up folder names for default values
		self.data_folder = od_path + 'Data'
		self.image_folder = self.data_folder + '/Source_Images'
		self.model_folder = self.data_folder + '/Model_Weights'
		self.model_weights = self.model_folder + '/trained_weights_final.h5'
		self.model_classes = self.model_folder + '/data_classes.txt'
		self.anchors_path = src_path + '/keras_yolo3/model_data/yolo_anchors.txt'
		FLAGS = None

		# define YOLO detector
		self.yolo = YOLO(
			**{
				"model_path": self.model_weights,
				"anchors_path": self.anchors_path,
				"classes_path": self.model_classes,
				"score": 0.25,
				"gpu_num": 1,
				"model_image_size": (416, 416),
			}
		)

	def detect(self, image_path):
		# prediction: [[xmin, ymin, xmax, ymax, classIndex, confidence], ...]
		prediction, image = detect_object(
				self.yolo,
				image_path,
				save_img=False,
				save_img_path=od_path+'/tmp',
				postfix='_od',
			)
		
		res = self.decode_predictions(prediction)
		print('prediction:', res, image.shape)
		return res, image.shape[1], image.shape[0]
	
	def decode_predictions(self, predictions):
		res = []
		class_names = self.yolo._get_class()
		for pred in predictions:
			res.append({'xmin':pred[0], 'ymin':pred[1], 'xmax':pred[2], 'ymax':pred[3], 'class':class_names[pred[4]], 'confidence':pred[5]})
		return res

if __name__ == '__main__':
    od = ObjectDetector()
    od.detect('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg')


