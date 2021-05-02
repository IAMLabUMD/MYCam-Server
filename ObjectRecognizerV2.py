'''
	A teachable object recognizer class with transfer learning

	Author: Jonggi Hong
	Date: 12/13/2020
'''

import time
import numpy as np
import os
import scipy.stats
import threading
import sys
import tarfile
import random
from datetime import datetime
from collections import OrderedDict, defaultdict
import glob
import hashlib
import os.path

import re
import csv
import json

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # https://github.com/tensorflow/tensorflow/issues/1258import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pathlib import Path
from shutil import copyfile, rmtree

from six.moves import urllib
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_shape
import subprocess

# set the seed of the random function for reproducability
random.seed(10)

def change_permissions_recursive(path, mode):
	for root, dirs, files in os.walk(path, topdown=False):
		for dir in [os.path.join(root,d) for d in dirs]:
			os.chmod(dir, mode)
	for file in [os.path.join(root, f) for f in files]:
			os.chmod(file, mode)




class ObjectRecognizer:
	def __init__(self):
		self.debug = False
		self.curr_model_dir = 'no model is loaded.'
		self.labels = None
		self.graph = None
		self.sess = None
		self.read_image_session = None
		
		# These are all parameters that are tied to the particular model architecture
		# we're using for Inception v3. These include things like tensor names and their
		# sizes. If you want to adapt this script to work with another model, you will
		# need to update these to reflect the values in the network you're using.
		# pylint: disable=line-too-long
		self.base_model_dir = '/home/jhong12/TOR-app-files/base_model'
		self.TRAIN_PHOTO_NUM = 30
		self.K_SHOT = 25
		self.MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
		
		self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
		# pylint: enable=line-too-long
		self.BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
		self.BOTTLENECK_TENSOR_SIZE = 2048
		self.MODEL_INPUT_WIDTH = 299
		self.MODEL_INPUT_HEIGHT = 299
		self.MODEL_INPUT_DEPTH = 3
		self.JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
		self.RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
		self.RANDOM_SEED_GRAPH = 37
		self.RANDOM_SEED = 12
		self.how_many_training_steps = 500
		self.final_tensor_name = 'final_result'
		self.learning_rate = 0.01
		self.train_batch_size = 100
		self.eval_step_interval = 50
		self.INPUT_MEAN = 128
		self.INPUT_STD = 128





	''' loads the classification model and labels
		
		Arguments:
			- model_dir: the directory with the model and label files
		
		Returns:
			- model: keras model instance from the model file
			- labels: list of labels
	'''
	def load_model_and_labels(self, model_dir):
		graph = tf.Graph()
		graph_def = tf.GraphDef()

		with open(os.path.join(model_dir, 'model.pb'), "rb") as f:
			graph_def.ParseFromString(f.read())
		with graph.as_default():
			tf.import_graph_def(graph_def)
		
		label = []
		proto_as_ascii_lines = tf.gfile.GFile(os.path.join(model_dir, 'labels.txt')).readlines()
		for l in proto_as_ascii_lines:
			label.append(l.rstrip())
		
		self.sess = tf.Session(graph=graph)
		self.graph = graph
		self.curr_model_dir = model_dir
		self.labels = label
		
		





	''' Use 'org_dir' parameter to load the model and labels to save in the 'save_dir' directory.

			Arguments:
				- model_dir: the directory to save the model and labels
				- org_dir: the directory with model and labels. 
			Returns:
	'''
	def save_model_and_labels(self, save_dir, org_dir):
		if self.debug:
			print('Saving the model...', save_dir)
		Path(save_dir).mkdir(parents=True, exist_ok=True)  # create the directory if it does not exist.

		if os.path.isdir(org_dir):
			org_model_path = os.path.join(org_dir, 'model.pb')
			org_labels_path = os.path.join(org_dir, 'labels.txt')
			
			if os.path.isfile(org_model_path):
				copyfile(org_model_path, os.path.join(save_dir, 'model.pb'))
			if os.path.isfile(org_labels_path):
				copyfile(org_labels_path, os.path.join(save_dir, 'labels.txt'))
		else:
			print('The model to save is not found (no previous model).')






	''' saves bottleneck features of the images
		https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
		
			Arguments:
				- img_dir: the directory with training samples
				- base_model: the base model used to create feature vectors of images
				- bottleneck_dir: the directory where the feature vectors and labels of the features will be saved
				
			Return:
				- features: feature vectors of the images
				- features_labels: labels of the feature vectors
				- labels: a directory with labels (index, label)
	'''
	def get_bottleneck_features(self, img_dir):
		train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
		train_generator = train_datagen.flow_from_directory(img_dir,
															target_size=(self.input_width, self.input_height),
															color_mode='rgb',
															batch_size=50,
															class_mode='categorical',
															shuffle=True)

		class_indices = (train_generator.class_indices)
		labels = dict((v, k) for k, v in class_indices.items())

		bottleneck_features = self.base_model.predict(train_generator)
		bottleneck_labels = train_generator.classes

		return bottleneck_features, bottleneck_labels, labels









	
	"""Download and extract model tar file.

		If the pretrained model we're using doesn't already exist, this function
		downloads it from the TensorFlow.org website and unpacks it into a directory.
	"""
	def maybe_download_and_extract(self):
		dest_directory = self.base_model_dir
		if not os.path.exists(dest_directory):
			os.makedirs(dest_directory)
			
		filename = self.DATA_URL.split('/')[-1]
		filepath = os.path.join(dest_directory, filename)
		if not os.path.exists(filepath):
			def _progress(count, block_size, total_size):
				sys.stdout.write('\r>> Downloading %s %.1f%%' %
							   (filename,
								float(count * block_size) / float(total_size) * 100.0))
				sys.stdout.flush()

			filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress)
			print()
			statinfo = os.stat(filepath)
			print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)






	""""Creates a graph from saved GraphDef file and returns a Graph object.

		Returns:
		Graph holding the trained Inception network, and various tensors we'll be
		manipulating.
	"""
	def create_inception_graph(self, attempt_number):
		with tf.Session() as sess:
			model_filename = os.path.join(self.base_model_dir, 'classify_image_graph_def.pb')
		with gfile.FastGFile(model_filename, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
				tf.import_graph_def(graph_def, name='', return_elements=[
					self.BOTTLENECK_TENSOR_NAME, self.JPEG_DATA_TENSOR_NAME,
					self.RESIZED_INPUT_TENSOR_NAME]))
			# Set seed for a deterministic behavior at the graph level
			tf.set_random_seed(self.RANDOM_SEED_GRAPH + attempt_number)
		return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor






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
	
	def create_image_lists(self, image_dir, training_number, testing_number, validation_number):
		if not tf.gfile.Exists(image_dir):
			tf.logging.error("Image directory '" + image_dir + "' not found.")
			return None
		result = OrderedDict()
		sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
		# The root directory comes first, so skip it.
		is_root_dir = True
		for sub_dir in sub_dirs:
			if is_root_dir:
				is_root_dir = False
				continue
			extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
			file_list = []
			dir_name = os.path.basename(sub_dir)
			if dir_name == image_dir:
				continue
			tf.logging.info("Looking for images in '" + dir_name + "'")
			for extension in extensions:
				file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
				file_list.extend(tf.gfile.Glob(file_glob))
			if not file_list:
				tf.logging.warning('No files found')
				continue
			if len(file_list) < 20:
				tf.logging.warning(
					'WARNING: Folder has less than 20 images, which may cause issues.')
			elif len(file_list) > self.MAX_NUM_IMAGES_PER_CLASS:
				tf.logging.warning(
					'WARNING: Folder {} has more than {} images. Some images will '
					'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
			label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
			training_images = []
			testing_images = []
			validation_images = []
			# Here simply randomize the file list to simulate
			# "the random selection of training, validation, and testing sets"
			random.shuffle(file_list)
			for file_name in file_list:
				base_name = os.path.basename(file_name)
				if len(training_images) < training_number:
					training_images.append(base_name)
				elif len(validation_images) < validation_number:
					validation_images.append(base_name)
				elif len(testing_images) < testing_number:
					testing_images.append(base_name)
				else:
					# no need to select images anymore
					break

			result[label_name] = {
				'dir': dir_name,
				'training': training_images,
				'testing': testing_images,
				'validation': validation_images,
			}
# 			print(label_name, len(training_images), len(validation_images), training_number, validation_number)
		return result




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
	def cache_bottlenecks(self, sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
		how_many_bottlenecks = 0
		self.ensure_dir_exists(bottleneck_dir)
		for label_name, label_lists in image_lists.items():
			for category in ['training', 'testing', 'validation']:
				try:
					category_list = label_lists[category]
					for index, unused_base_name in enumerate(category_list):
						self.get_or_create_bottleneck(sess, image_lists, label_name, index,
												image_dir, category, bottleneck_dir,
												jpeg_data_tensor, bottleneck_tensor)
					how_many_bottlenecks += 1
					if how_many_bottlenecks % 100 == 0:
						print(str(how_many_bottlenecks) + ' bottleneck files created.')
						
				except KeyError:
					print("No files, skipping")
					continue





	"""Runs inference on an image to extract the 'bottleneck' summary layer.
		Args:
		sess: Current active TensorFlow Session.
		image_data: Numpy array of image data.
		image_data_tensor: Input data layer in the graph.
		bottleneck_tensor: Layer before the final softmax.

		Returns:
		Numpy array of bottleneck values.
	"""
	def run_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
		bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
		bottleneck_values = np.squeeze(bottleneck_values)
		return bottleneck_values




	"""Makes sure the folder exists on disk.

		Args:
		dir_name: Path string to the folder we want to create.
	"""
	def ensure_dir_exists(self, dir_name):
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
			subprocess.call(['chmod', '0777', dir_name])
# 			change_permissions_recursive(dir_name, 0o777)





		"""Returns a path to an image for a label at the given index.

		Args:
		  image_lists: OrderedDict of training images for each label.
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
	def get_image_path(self, image_lists, label_name, index, image_dir, category):
		if label_name not in image_lists:
			tf.logging.fatal('Label does not exist %s.', label_name)
		label_lists = image_lists[label_name]
		if category not in label_lists:
			tf.logging.fatal('Category does not exist %s.', category)
		category_list = label_lists[category]
		if not category_list:
			tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
		mod_index = index % len(category_list)
		base_name = category_list[mod_index]
		sub_dir = label_lists['dir']
		full_path = os.path.join(image_dir, sub_dir, base_name)
		return full_path





	"""Returns a path to a bottleneck file for a label at the given index.

		Args:
		  image_lists: OrderedDict of training images for each label.
		  label_name: Label string we want to get an image for.
		  index: Integer offset of the image we want. This will be moduloed by the
		  available number of images for the label, so it can be arbitrarily large.
		  bottleneck_dir: Folder string holding cached files of bottleneck values.
		  category: Name string of set to pull images from - training, testing, or
		  validation.
		  module_name: The name of the image module being used.

		Returns:
		  File system path string to an image that meets the requested parameters.
		"""
	def get_bottleneck_path(self, image_lists, label_name, index, bottleneck_dir, category):
		return self.get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'






	"""Runs inference on an image to extract the 'bottleneck' summary layer.

		Args:
		sess: Current active TensorFlow Session.
		image_data: Numpy array of image data.
		image_data_tensor: Input data layer in the graph.
		bottleneck_tensor: Layer before the final softmax.

		Returns:
		Numpy array of bottleneck values.
	"""
	def run_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
		bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
		bottleneck_values = np.squeeze(bottleneck_values)
		return bottleneck_values






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
	def get_or_create_bottleneck(self, sess, image_lists, label_name, index, image_dir,
								 category, bottleneck_dir, jpeg_data_tensor,
								 bottleneck_tensor):
		label_lists = image_lists[label_name]
		sub_dir = label_lists['dir']
		sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
		self.ensure_dir_exists(sub_dir_path)
		bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index,
											bottleneck_dir, category)
		if not os.path.exists(bottleneck_path):
			print('Creating bottleneck at ' + bottleneck_path)
			image_path = self.get_image_path(image_lists, label_name, index, image_dir, category)
			if not gfile.Exists(image_path):
				tf.logging.fatal('File does not exist %s', image_path)
			image_data = gfile.FastGFile(image_path, 'rb').read()
			bottleneck_values = self.run_bottleneck_on_image(sess, image_data,
														jpeg_data_tensor,
														bottleneck_tensor)
			bottleneck_string = ','.join(str(x) for x in bottleneck_values)
			with open(bottleneck_path, 'w') as bottleneck_file:
				bottleneck_file.write(bottleneck_string)

		with open(bottleneck_path, 'r') as bottleneck_file:
			bottleneck_string = bottleneck_file.read()
		bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
		return bottleneck_values







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
	def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor):
		bottleneck_input = tf.placeholder_with_default(
			bottleneck_tensor, shape=[None, self.BOTTLENECK_TENSOR_SIZE],
			name='BottleneckInputPlaceholder')
		layer_weights = tf.Variable(
			tf.truncated_normal([self.BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
			name='final_weights')
		layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
		logits = tf.matmul(bottleneck_input, layer_weights,
						 name='final_matmul') + layer_biases
		final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
		ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
			cross_entropy_mean)
		return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
			final_tensor, layer_weights, layer_biases)



	"""Inserts the operations we need to evaluate the accuracy of our results.

	Args:
		result_tensor: The new final node that produces results.
		ground_truth_tensor: The node we feed ground truth data
		into.

	Returns:
		Nothing.
	"""

	def add_evaluation_step(self, result_tensor, ground_truth_tensor):
		correct_prediction = tf.equal(
			tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
		evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
		return evaluation_step






	"""Inserts the operations we need to evaluate the accuracy of our results.

	Args:
		result_tensor: The new final node that produces results.
		ground_truth_tensor: The node we feed ground truth data
		into.

	Returns:
		Nothing.
	"""

	def add_correct_prediction(self, result_tensor, ground_truth_tensor):
		correct_prediction = tf.equal(
			tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
		return correct_prediction







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

	def get_random_cached_bottlenecks(self, sess, image_lists, how_many, category,
									bottleneck_dir, image_dir, jpeg_data_tensor,
									bottleneck_tensor):
		class_count = len(image_lists.keys())
		bottlenecks = []
		ground_truths = []
		for unused_i in range(how_many):
			label_index = random.randrange(class_count)
			label_name = list(image_lists.keys())[label_index]
			# print(label_name)
			image_index = random.randrange(65536)
			bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,
												image_index, image_dir, category,
												bottleneck_dir, jpeg_data_tensor,
												bottleneck_tensor)
			ground_truth = np.zeros(class_count, dtype=np.float32)
			ground_truth[label_index] = 1.0
			bottlenecks.append(bottleneck)
			ground_truths.append(ground_truth)
		return bottlenecks, ground_truths






	"""Insert the operations to calculate the precision per object

		Args:
		result_tensor: The new final node that produces results.
		ground_truth_tensor: The node we feed ground truth data into.

		Returns:
		a list of precision_per_object
		"""
	def add_evaluation_per_object(self, result_tensor, ground_truth_tensor, num_classes):
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





	''' trains the object recognition model with the bottleneck features and saves the model and labels to files.
		The bottleneck features are used to train the model faster.
		
		ATTENTION: This function is not implemented completely yet. This does not train a model correctly.

			Arguments:
				- model_dir: the directory to save the model and labels
				- img_dir: the directory with training samples (images)

			Returns:

	'''
	def train_with_bottleneck(self, model_dir, img_dir):
		if not self.sess is None:
			self.sess.close()
		tf.reset_default_graph()
		
		start = time.time()
		
		self.start_time = time.time()
		if self.debug:
			print('Start training ...', img_dir)
		
		# Set up the pre-trained graph.
		self.maybe_download_and_extract()
		graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = \
			self.create_inception_graph(1)
		
		# Look at the folder structure, and create lists of all the images.
		image_lists = self.create_image_lists(img_dir, self.K_SHOT, 0, 
						self.TRAIN_PHOTO_NUM-self.K_SHOT)
		class_count = len(image_lists.keys())
		if class_count == 0:
			print('No valid folders of images found at ' + img_dir)
			return -1
		if class_count < 3:
			print('Only one valid folder of images found at ' + img_dir +
				' - multiple classes are needed for classification.')
			return -1
				
		sess = tf.Session()
		
		# We'll make sure we've calculated the 'bottleneck' image summaries and
		# cached them on disk.
		bottleneck_dir = os.path.join(model_dir, 'bottlenecks')
		if not os.path.isdir(bottleneck_dir):
			Path(bottleneck_dir).mkdir(parents=True, exist_ok=True)
			subprocess.call(['chmod', '0777', bottleneck_dir])
			
		self.cache_bottlenecks(sess, image_lists, img_dir, bottleneck_dir, jpeg_data_tensor, 
							bottleneck_tensor)
		
		# Add the new layer that we'll be training.
		(train_step, cross_entropy, bottleneck_input, ground_truth_input,
		final_tensor, layer_weights, layer_biases) = self.add_final_training_ops(len(image_lists.keys()),
														self.final_tensor_name,
														bottleneck_tensor)

		# Create the operations we need to evaluate the accuracy of our new layer.
		evaluation_step = self.add_evaluation_step(final_tensor, ground_truth_input)
		correct_prediction_step = self.add_correct_prediction(final_tensor, ground_truth_input)
		(precisions_step, recalls_step) = self.add_evaluation_per_object(final_tensor,
																ground_truth_input,
																class_count)

		# Set up all our weights to their initial default values.
		init = tf.global_variables_initializer()
		sess.run(init)
		sess.run(tf.local_variables_initializer())
		
		start_training = datetime.now()
		# Run the training for as many cycles as requested on the command line.
		for i in range(self.how_many_training_steps):
			# Get a catch of input bottleneck values, either calculated fresh every time
			# with distortions applied, or from the cache stored on disk.
			train_bottlenecks, train_ground_truth = self.get_random_cached_bottlenecks(
				sess, image_lists, self.train_batch_size, 'training',
				bottleneck_dir, img_dir, jpeg_data_tensor,
				bottleneck_tensor)
			# Feed the bottlenecks and ground truth into the graph, and run a training
			# step.
			sess.run(train_step,
					 feed_dict={bottleneck_input: train_bottlenecks,
								ground_truth_input: train_ground_truth})
			# Every so often, print out how well the graph is training.
			is_last_step = (i + 1 == self.how_many_training_steps)
			if (i % self.eval_step_interval) == 0 or is_last_step:
				train_accuracy, cross_entropy_value = sess.run(
					[evaluation_step, cross_entropy],
					feed_dict={bottleneck_input: train_bottlenecks,
							 ground_truth_input: train_ground_truth})
				print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
																train_accuracy * 100))
				print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
														 cross_entropy_value))
			
			if self.TRAIN_PHOTO_NUM-self.K_SHOT > 0:
				validation_bottlenecks, validation_ground_truth = (
					self.get_random_cached_bottlenecks(
						sess, image_lists, self.TRAIN_PHOTO_NUM-self.K_SHOT, 'validation',
						bottleneck_dir, img_dir, jpeg_data_tensor,
						bottleneck_tensor))
				validation_accuracy = sess.run(
					evaluation_step,
					feed_dict={bottleneck_input: validation_bottlenecks,
							ground_truth_input: validation_ground_truth})
				print('%s: Step %d: Validation accuracy = %.1f%%' %
					(datetime.now(), i, validation_accuracy * 100))
		
		# Write out the trained graph and labels with the weights stored as constants.
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, sess.graph.as_graph_def(), [self.final_tensor_name])
		with tf.gfile.FastGFile(model_dir+'/model.pb', 'wb') as f:
			f.write(output_graph_def.SerializeToString())
		with tf.gfile.FastGFile(model_dir+'/labels.txt', 'w') as f:
			f.write('\n'.join(image_lists.keys()) + '\n')
			
		self.load_model_and_labels(model_dir)

		end=time.time()
		print('\nTraining time: {:.3f}s\n'.format(end-start))
		return 1
	
	
	
	
	


	def read_tensor_from_image_file(self, file_name, input_height=299, input_width=299,
									input_mean=0, input_std=255):
		input_name = "file_reader"
		output_name = "normalized"
		
		read_tensor_graph = tf.Graph()
		with read_tensor_graph.as_default():
			file_reader = tf.read_file(file_name, input_name)
			if file_name.endswith(".png"):
				image_reader = tf.image.decode_png(file_reader, channels = 3,
											name='png_reader')
			elif file_name.endswith(".gif"):
				image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
														name='gif_reader'))
			elif file_name.endswith(".bmp"):
				image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
			else:
				image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
												name='jpeg_reader')
	
			float_caster = tf.cast(image_reader, tf.float32)
			dims_expander = tf.expand_dims(float_caster, 0);
			resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
			normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
		
			with tf.Session(graph=read_tensor_graph) as read_image_session:
				result = read_image_session.run(normalized)
				return result

	
	
			
	

	''' predicts the object in an image. This function uses the model trained with the bottleneck features (train_with_bottleneck).
	
		ATTENTION: The train_with_bottleneck() function is not implemented yet. This function can be used after train_with_bottleneck()
		function is implemented.

			Arguments:
				- model_dir: the directory with the model and labels
				- img_path: the target image

			Returns:
				- best_label: the label with the highest confidence score
				- entropy: entropy of the confidence scores
				- conf: a dictionary with confidence scores of all labels (label, confidence score)
	'''
	def predict_with_bottleneck(self, model_dir, img_path):
		# if the model does not exist, return None
		if not os.path.isdir(model_dir):
			return None, None, None

		if self.curr_model_dir != model_dir:
			self.load_model_and_labels(model_dir)
			print('load the graph')

		t = self.read_tensor_from_image_file(img_path,
									input_height=self.MODEL_INPUT_HEIGHT,
									input_width=self.MODEL_INPUT_WIDTH,
									input_mean=self.INPUT_MEAN,
									input_std=self.INPUT_STD)
		input_layer = "Mul"
		output_layer = "final_result"
		
		with self.graph.as_default():

			input_name = "import/" + input_layer
			output_name = "import/" + output_layer
			input_operation = self.graph.get_operation_by_name(input_name)
			output_operation = self.graph.get_operation_by_name(output_name)

			start = time.time()
			results = self.sess.run(output_operation.outputs[0],
							  {input_operation.outputs[0]: t})
			end=time.time()
			
			results = np.squeeze(results)
			top_k = results.argsort()[::-1]

			print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

			conf = {}
			best_label = ''
			max_prob = 0
			for i in top_k:
				conf[self.labels[i]] = results[i]
				if results[i] > max_prob:
					best_label = self.labels[i]
					max_prob = results[i]
				
			entropy = scipy.stats.entropy(results)
			return best_label, entropy, conf





	def printSessionInfo(self, indicator):
		print('!!!!!! session graph', indicator, len(tf.get_default_graph().get_operations()), [n.name for n in tf.get_default_graph().get_operations()[:3]+tf.get_default_graph().get_operations()[-3:]])
		if not self.graph is None:
			print('!!!!!! graph instance', indicator, len(self.graph.get_operations()), [n.name for n in self.graph.get_operations()[:3]+self.graph.get_operations()[-5:]])
		else:
			print('!!!!!! graph instance is None')
# 		for op in self.graph.get_operations():
# 			if 'import' in op.name:
# 				print('op name', op.name)
		
		
		
		
		
		
	def train(self, model_dir, img_dir):
		train_res = self.train_with_bottleneck(model_dir, img_dir)
		subprocess.call(['chmod', '0777', model_dir+'/labels.txt'])
		self.predict(model_dir, '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Remote/1.jpg') # warm up
		return train_res
		
	
	def train_with_steps(self, model_dir, img_dir, steps):
		self.how_many_training_steps = steps
		self.train(model_dir, img_dir)
		
		
	def predict(self, model_dir, img_path):
		return self.predict_with_bottleneck(model_dir, img_path)
	
	
	
	
	def reset(self, model_dir):
		try:
			rmtree(model_dir)
		except OSError as e:
			print("Reset error: %s : %s" % (model_dir, e.strerror))

if __name__ == '__main__':
	# test codes
	orec = ObjectRecognizer()
	orec.debug = True
# 	orec.train_with_steps('/home/jhong12/URCam/model', '/home/jhong12/TOR-app-files/DatasetForURCam/Images', 1000)
# 	print(orec.predict('/home/jhong12/GORTestModels/tmp', '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Remote/1.jpg')) # warm up
	orec.train('/home/jhong12/TOR-app-files/models/AFE75DDE-6684-4652-9688-76B3AFBA90D6', '/home/jhong12/TOR-app-files/photo/TrainFiles/AFE75DDE-6684-4652-9688-76B3AFBA90D6') # Ebrima's second pilot data
