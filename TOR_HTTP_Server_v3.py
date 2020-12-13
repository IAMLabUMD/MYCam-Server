from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from shutil import copyfile

import argparse
import scipy.stats
import sys
import os
import traceback

import time
import csv
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Flatten
from keras.applications import MobileNet
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import f1_score
from numpy import asarray
from numpy import savetxt
from pathlib import Path


#     from keras.applications.inceptionresnetv2 import preprocess_input
#     from keras.applications.inceptionresnetv2 import decode_predictions
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

prevTestID = ""
model_file = ""
label_file = ""
input_height = 299
input_width = 299
input_mean = 128
input_std = 128
# input_layer = "input"
input_layer = "Mul"
output_layer = "final_result"

model = None

input_name = ""
output_name = ""
input_operation = None
output_operation = None
isTraining = False

os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu

# ssh jhong12@128.8.235.4
# sudo lsof -i -P -n | grep LISTEN
# nvidia-smi
# source ./venv/bin/activate  
# run virtual env
# nohup python3 -u TOR_HTTP_Server_v3.py &> server_log &
# ps -aux
# sudo ps -U jhong12

# ssh -N -f -L localhost:4000:localhost:4000 jhong12@128.8.235.4

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
	def safeGetValue(self, dic, k):
		if k in dic:
			return dic[k]
		else:
			return None

	def parseParams(self):
		content_length = int(self.headers['Content-Length'])
		html_body = self.rfile.read(content_length)

		postStr = html_body.decode().replace("%2F", "/")
		postStr = postStr.split("&")
		params = {}
		for pstr in postStr:
			pstr = pstr.split("=")
			params[pstr[0]] = pstr[1]
			
		print(params)
		userID = self.safeGetValue(params, 'userID')
		cmd = self.safeGetValue(params, 'type')
		category = self.safeGetValue(params, 'category')
		img_path = self.safeGetValue(params, 'imgPath')
		return userID, cmd, category, img_path
		

	def recognize(self, imgPath, userID):
		print(prevTestID, "....")
# 		imgPath = params["imgPath"]

		if self.prevTestID != userID:
			print("different id")
			model_file = "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+".h5"
			label_file = "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+".txt"
			
			if not os.path.isfile(model_file):
				output = "Error. Model does not exist."
				response.write(str.encode(output))
		
			model = load_model(model_file)
		else:
			print("same id")
		
		img = image.load_img(imgPath, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis = 0)
		x = preprocess_input(x)

		preds = model.predict(x)[0]
		print(preds)
	
		labels = []			
		with open(label_file) as openfileobject:
			for line in openfileobject:
				labels.append(line.strip())
	
		entropy = scipy.stats.entropy(preds)
		output = "" + str(entropy)
		for i in range(len(labels)):
			output = output + "-:-" + str(labels[i])+"/"+str(preds[i])
		
		print(output)
		response.write(str.encode(output))
		

	def do_GET(self):
		print("GET")
		self.send_response(200)
		self.end_headers()
		self.wfile.write(b'Hello, world!')

	def do_POST(self):
		try:
			global prevTestID, model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, model, input_name, output_name, input_operation, output_operation, isTraining
			start_time = time.time()
# 			content_length = int(self.headers['Content-Length'])
# 			body = self.rfile.read(content_length)
# 
# 			postStr = body.decode().replace("%2F", "/")
# 			postStr = postStr.split("&")
# 			params = {}
# 			for pstr in postStr:
# 				pstr = pstr.split("=")
# 				params[pstr[0]] = pstr[1]
# 		
# 		
# 			print(params)
# 			userID = params["userId"]
# 			type = params["type"]
# 			category = params["category"]
			userID, cmd, category, img_path = self.parseParams()
			response = BytesIO()
		
			if cmd == "test":
				print(prevTestID, "....")
				imgPath = params["imgPath"]

				if prevTestID != userID:
					print("different id")
					model_file = "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+".h5"
					label_file = "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+".txt"
					
					if not os.path.isfile(model_file):
						output = "Error. Model does not exist."
						response.write(str.encode(output))
				
					model = load_model(model_file)
				else:
					print("same id")
				
			

				img = image.load_img(imgPath, target_size=(224, 224))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis = 0)
				x = preprocess_input(x)
 
				preds = model.predict(x)[0]
				print(preds)
			
				labels = []			
				with open(label_file) as openfileobject:
					for line in openfileobject:
						labels.append(line.strip())
			
				entropy = scipy.stats.entropy(preds)
				output = "" + str(entropy)
				for i in range(len(labels)):
					output = output + "-:-" + str(labels[i])+"/"+str(preds[i])
				
				print(output)
				response.write(str.encode(output))
			
			elif cmd == "loadModel":
				model_file = "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+".h5"
				label_file = "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+".txt"
			
				model = load_model(model_file)
			
			elif cmd == "saveTrainPhoto":
				imgPath = params["imgPath"]
			
			
			elif cmd == "trainRequest":	
				markFile = "/home/jhong12/TOR-app-files/isTraining"
				f = open(markFile, "w")
				f.write("yes")
				f.close()
			
				Path("/home/jhong12/TOR-app-files/models/"+userID).mkdir(parents=True, exist_ok=True)
				os.chmod("/home/jhong12/TOR-app-files/models/"+userID, 0o777)
			
				model_file = "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+".h5"
				label_file = "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+".txt"
				model_prev_file = "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+"_prev.h5"
				label_prev_file = "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+"_prev.txt"
			
				if os.path.isfile(model_file):
					copyfile(model_file, "/home/jhong12/TOR-app-files/models/"+userID+"/model_"+category+"_prev.h5")
				if os.path.isfile(label_file):
					copyfile(label_file, "/home/jhong12/TOR-app-files/models/"+userID+"/label_"+category+"_prev.txt")
			
				print('preprocessing time: ', time.time()-start_time)	
			
				train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
				train_path = "/home/jhong12/TOR-app-files/photo/TrainFiles/"+userID+"/"+category
				print('dir:', train_path)
				train_generator=train_datagen.flow_from_directory(train_path,
																 target_size=(224,224),
																 color_mode='rgb',
																 batch_size=30,
																 class_mode='categorical',
																 shuffle=True)
															
				labels = (train_generator.class_indices)
				labels2 = dict((v,k) for k,v in labels.items())
				print('collecting training data time: ', time.time()-start_time)	
			 
				base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(224,224,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
	
				print('loading a model time: ', time.time()-start_time)
				x=base_model.output
				x = Flatten()(x)
				preds=Dense(len(labels2),activation='softmax')(x) #final layer with softmax activation
	
				model=Model(inputs=base_model.input,outputs=preds)
				model.layers[-1].trainable = True												 
			
				model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
			
				print('setting up the training time: ', time.time()-start_time)
			
				step_size_train=train_generator.n//train_generator.batch_size
				model.fit_generator(generator=train_generator,
								   steps_per_epoch=step_size_train,
								   epochs=100) # 200? # 80: around 2 minutes, 200: around 5 minutes
							   
				print('training done time: ', time.time()-start_time)
				model.save(model_file)
			
				print('save model time: ', time.time()-start_time)
				# write labels
			
				f = open(label_file, "w")
				for i in range(len(labels2)):
					f.write(labels2[i]+"\n")
				f.close()
				os.chmod(label_file, 0o777)
				os.chmod(model_file, 0o777)
				os.chmod(label_prev_file, 0o777)
				os.chmod(model_prev_file, 0o777)
			
				f = open(markFile, "w")
				f.write("no")
				f.close()
			
				print('print response: training is done.')
				response.write(b'Training is done')
			else:
				print("Debugging...")
			

	# 		self.send_response(200)
	# 		self.end_headers()
	# 		response.write(b'This is POST request. ')
	# 		response.write(b'Received: ')
	# 		response.write(body)
			self.wfile.write(response.getvalue())
		except:
			print("Exception. Reset training.")
			markFile = "/home/jhong12/TOR-app-files/isTraining"
			f = open(markFile, "w")
			f.write("no")
			f.close()

			print(traceback.format_exc())
			
			e = sys.exc_info()[0]
			msg = "Error: %s" % e
			self.wfile.write(msg.encode())
		
		prevTestID = userID

        
        


print("run")
markFile = "/home/jhong12/TOR-app-files/isTraining"
f = open(markFile, "w")
f.write("no")
f.close()
			
httpd = HTTPServer(('128.8.235.4', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
