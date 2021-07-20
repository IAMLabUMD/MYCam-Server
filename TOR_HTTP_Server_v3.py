'''
    An HTTP server that trains the object recognizer model and gets a prediction from it remotely.

    Author: Jonggi Hong
    Date: 12/13/2020
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

import sys
import traceback

import time
import os
from ObjectRecognizerV2 import ObjectRecognizer
from DescriptorGenerator import DescriptorGenerator
from StudyHelper import StudyHelper
from datetime import datetime



# ssh jhong12@128.8.235.4
# sudo lsof -i -P -n | grep LISTEN
# nvidia-smi
# source ./venv/bin/activate  
# run virtual env
# nohup python3 -u TOR_HTTP_Server_v3.py &> server_log &
# ps -aux
# sudo ps -U jhong12

# jupyter notebook --no-browser --port=4000
# ssh -N -f -L localhost:4000:localhost:4000 jhong12@128.8.235.4
# Current PID: 1q

log_path = '../logs/request_logs.txt'
		
def writeLog(line):
	# datetime object containing current date and time
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	f= open(log_path,"a+")
	f.write(line+','+dt_string+'\n')
	f.close()

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
	def safeGetValue(self, dic, k):
		if k in dic:
			return dic[k]
		else:
			return 'None'

	def parseParams(self):
		content_length = int(self.headers['Content-Length'])
		html_body = self.rfile.read(content_length)

		postStr = html_body.decode().replace('%2F', '/').replace('+', ' ')
		postStr = postStr.split('&')
		params = {}
		for pstr in postStr:
			pstr = pstr.split('=')
			params[pstr[0]] = pstr[1]

		print(params)
		userID = self.safeGetValue(params, 'userId')
		cmd = self.safeGetValue(params, 'type')
		category = self.safeGetValue(params, 'category')
		img_path = self.safeGetValue(params, 'imgPath')
		request_time = self.safeGetValue(params, 'time')
		org_fname = self.safeGetValue(params, 'org_fname')
		object_name = self.safeGetValue(params, 'object_name')
		return userID, cmd, category, img_path, request_time, org_fname, object_name

	def do_GET(self):
		try:
			print('GET')
			self.send_response(200)
			self.end_headers()
			self.wfile.write(b'Hello, world!')
		except:
			print('GET Exception')
			print(traceback.format_exc())

	def do_POST(self):
		global object_recognizer, des_generator
	
		try:
			start_time = time.time()
			userID, cmd, category, img_path, request_time, org_fname, object_name = self.parseParams()
			model_dir = '/home/jhong12/TOR-app-files/models/' + userID
			print(userID, cmd, category, img_path)
			writeLog('request,'+userID+','+cmd+','+category+','+img_path+','+request_time+','+org_fname+','+object_name)
			response = BytesIO()

			if cmd == 'test':
				best_label, entropy, conf = object_recognizer.predict(model_dir, img_path)

				output = ''
				if best_label is None: # if the model does not exist
					output = 'Object recognition model does not exist.'
				else:
					output = str(entropy)
					for label, confidence in conf.items():
						output = output + '-:-' + label + '/' + str(confidence)
				
				print(output)
				response.write(str.encode(output))
				writeLog('testResult,'+userID+','+org_fname+','+output)
			elif cmd == 'test-URCam':
				urcam_model_dir = '/home/jhong12/URCam/model'
				best_label, entropy, conf = object_recognizer.predict(urcam_model_dir, img_path)

				output = ''
				if best_label is None: # if the model does not exist
					output = 'Object recognition model does not exist.'
				else:
					output = str(entropy)
					for label, confidence in conf.items():
						output = output + '-:-' + label + '/' + str(confidence)
			
				print(output)
				response.write(str.encode(output))
				writeLog('testResult,'+userID+','+org_fname+','+output)
				

			elif cmd == 'loadModel':
				object_recognizer.load_model_and_labels(model_dir)

			elif cmd == 'trainRequest':
				markFile = '/home/jhong12/TOR-app-files/isTraining'
				f = open(markFile, 'w')
				f.write('yes')
				f.close()
				train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/' + userID

				if os.path.isdir(model_dir):
					object_recognizer.save_model_and_labels(model_dir + '_prev', org_dir=model_dir)
					sh = StudyHelper()
					sh.AddModelHistory(userID, model_dir, 'Train')
					
				train_res = object_recognizer.train(model_dir, train_img_dir)
				des_generator.initialize()

				f = open(markFile, 'w')
				f.write('no')
				f.close()

				if train_res == 1:
					print('Print response: training is done.')
					writeLog('trainingDone,'+userID)
					response.write(b'Training is done')
				else:
					print('Training failed: fewer than three objects.')
					writeLog('TrainingFail-FewerThanThree,'+userID)
					response.write(b'Training failed')
					
			elif cmd == 'getImgDescriptor':
				hand, blurry, cropped, small, desc_dic = des_generator.getImageDescriptor(img_path)
				
				print()
				print('##### Image Descriptors', org_fname)
				print('hand', hand, desc_dic['hand_area'])
				print('blurry', blurry, desc_dic['blurriness'])
				print('cropped', cropped, len(desc_dic['boxes']))
				print('small', small, len(desc_dic['boxes']))
				print()
				
				output = str(hand)+'#'+str(blurry)+'#'+str(cropped)+'#'+str(small)+'#'+str(desc_dic)+'#'+str(request_time)
				writeLog('imgDescriptors,'+userID+','+output+','+org_fname+','+str(desc_dic))
				response.write(str.encode(output))
				
			elif cmd == 'getSetDescriptor':
# 				train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/' + userID + '/' + category + '/' + object_name
				arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/' + userID + '/TrainedObjects/' + object_name + '-desc_info.txt'
				
				bg_var, side_var, dist_var, hand, blurry, cropped, small = des_generator.getSetDescriptor(arinfo_path)
				output = str(bg_var)+','+str(side_var)+','+str(dist_var)+','+str(hand)+','+str(blurry)+','+str(cropped)+','+str(small)
				writeLog('setDescriptors,'+userID+','+output)
				response.write(str.encode(output))
			elif cmd == 'getSetDescriptorForReview':
				arinfo_path = '/home/jhong12/TOR-app-files/ARInfo/' + userID + '/Temp/' + object_name
				
				bg_var, side_var, dist_var, hand, blurry, cropped, small = des_generator.getSetDescriptor(arinfo_path)
				output = str(bg_var)+','+str(side_var)+','+str(dist_var)+','+str(hand)+','+str(blurry)+','+str(cropped)+','+str(small)
				writeLog('getSetDescriptorForReview,'+userID+','+output)
				response.write(str.encode(output))
			elif cmd == 'Reset':
				if os.path.isdir(model_dir):
					sh = StudyHelper()
					save_name = sh.AddModelHistory(userID, model_dir, 'Reset')
					object_recognizer.reset(model_dir)
					response.write(str.encode('Reset:'+save_name))
				else:
					response.write(str.encode('No model to reset.'))
			elif cmd == 'rename':
				object_recognizer.load_model_and_labels(model_dir)
			else:
				print('Debugging...')

			self.wfile.write(response.getvalue())
		except:
			print('Exception. Reset training.')
			markFile = '/home/jhong12/TOR-app-files/isTraining'
			f = open(markFile, 'w')
			f.write('no')
			f.close()

			print(traceback.format_exc())

			e = sys.exc_info()[0]
			msg = 'Error: %s' % e
			self.wfile.write(msg.encode())


if __name__ == '__main__':
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="7"
	
	markFile = '/home/jhong12/TOR-app-files/isTraining'
	f = open(markFile, 'w')
	f.write('no')
	f.close()

	object_recognizer = ObjectRecognizer()
	object_recognizer.debug = True

	des_generator = DescriptorGenerator()
	des_generator.initialize()
	
	# generate descriptor for a dummy image to load models on the memory
	des_generator.getImageDescriptor('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg') # warm up

	print('run')

	httpd = HTTPServer(('127.0.0.1', 8000), SimpleHTTPRequestHandler)
	httpd.serve_forever()
