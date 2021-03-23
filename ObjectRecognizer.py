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
import tensorflow

from pathlib import Path
from shutil import copyfile, rmtree

from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical
from tensorflow.python.keras.backend import set_session



class ObjectRecognizer:
    def __init__(self):
        self.debug = False
        self.input_width = 224
        self.input_height = 224
        self.curr_model_dir = 'no model is loaded.'
        self.model = None
        self.labels = None
        
        self.sess = tensorflow.python.keras.backend.get_session()
        
        ####
        ## The base_model is used for train_with_bottleneck and predict_with_bottleneck functions,
        ## but these functions are not used now.
        ####
        # imports the mobilenet model and discards the last 1000 neuron layer.
        self.base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(self.input_width, self.input_height, 3))
        self.base_model.trainable = False
        print('loading the base model (', self.base_model.name, '): ')

    ''' loads the classification model and labels
        
        Arguments:
            - model_dir: the directory with the model and label files
        
        Returns:
            - model: keras model instance from the model file
            - labels: list of labels
    '''
    def load_model_and_labels(self, model_dir):
        if self.curr_model_dir == model_dir:
            return self.model, self.labels
        
        

        model_path = os.path.join(model_dir, 'model.h5')
        model = load_model(model_path)

        label_path = os.path.join(model_dir, 'labels.txt')
        labels = []
        with open(label_path) as openfileobject:
            for line in openfileobject:
                labels.append(line.strip())

        return model, labels

    ''' saves the current model and labels to file. When the model is not initialized, 
        use 'org_dir' parameter to load the model and labels to save in the 'save_dir' directory.

            Arguments:
                - model_dir: the directory to save the model and labels
                - org_dir: the directory with model and labels. 
            Returns:
    '''
    def save_model_and_labels(self, save_dir, org_dir=None):
        if self.debug:
            print('Saving the model...', save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)  # create the directory if it does not exist.

        # when the original files are given, just copy files.
        if not org_dir is None:
            if os.path.isdir(org_dir):
                copyfile(os.path.join(org_dir, 'model.h5'), os.path.join(save_dir, 'model.h5'))
                copyfile(os.path.join(org_dir, 'labels.txt'), os.path.join(save_dir, 'labels.txt'))
                self.model, self.labels = self.load_model_and_labels(org_dir)
            else:
                print('The model to save is not found (no previous model).')
            return

        # save the model to file
        model_path = os.path.join(save_dir, 'model.h5')
        set_session(self.sess)
        self.model.save(model_path)

        # save the labels to file
        label_path = os.path.join(save_dir, 'labels.txt')
        f = open(label_path, 'w')
        for i in range(len(self.labels)):
            f.write(self.labels[i] + '\n')
        f.close()

        if self.debug:
            print('saving the model to a file: ', time.time() - self.start_time)

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

    ''' trains the object recognition model and saves the model and labels to files

            Arguments:
                - model_dir: the directory to save the model and labels
                - img_dir: the directory with training samples (images)

            Returns:
            
    '''
    def train_without_bottleneck(self, model_dir, img_dir):
        if self.debug:
            print('Start training ...', img_dir)

        self.start_time = time.time()
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
        train_generator = train_datagen.flow_from_directory(img_dir,
                                                            target_size=(self.input_width, self.input_height),
                                                            color_mode='rgb',
                                                            batch_size=50,
                                                            class_mode='categorical',
                                                            shuffle=True)

        labels = (train_generator.class_indices)
        labels2 = dict((v, k) for k, v in labels.items())
        if self.debug:
            print('Training data are collected: ', time.time() - self.start_time)
            
        if len(labels) < 3:
            print('Cannot start training with fewer than three objects.', len(labels))
            return -1 # fail

        # imports the mobilenet model and discards the last 1000 neuron layer.
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(self.input_width, self.input_height, 3))
        base_model.trainable = False

        if self.debug:
            print('Loading the base model (', base_model.name, '): ', time.time() - self.start_time)

        x = base_model.output
        x = Flatten()(x)
        preds = Dense(len(labels2), activation='softmax')(x)  # final layer with softmax activation

        model = Model(inputs=base_model.input, outputs=preds)
        model.layers[-1].trainable = True

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if self.debug:
            print('Setting up the training: ', time.time() - self.start_time)

        step_size_train = train_generator.n // train_generator.batch_size
        model.fit(train_generator, steps_per_epoch=step_size_train,
                  epochs=50)  # 200? # 80: around 2 minutes, 200: around 5 minutes, 100: current

        if self.debug:
            print('Training is done: ', time.time() - self.start_time)

        # set current model and labels
        self.curr_model_dir = model_dir
        self.labels = labels2
        self.model = model

        # save the trained model and labels
        t = threading.Thread(target=self.save_model_and_labels, args=(model_dir,))
        t.start()
#         self.save_model_and_labels(model_dir)
        return 1 # success

    ''' predicts the object in an image

            Arguments:
                - model_dir: the directory with the model and labels
                - img_path: the target image

            Returns:
                - best_label: the label with the highest confidence score
                - entropy: entropy of the confidence scores
                - conf: a dictionary with confidence scores of all labels (label, confidence score)
    '''
    def predict_without_bottleneck(self, model_dir, img_path):
        # if the model does not exist, return None
        if not os.path.isdir(model_dir):
            return None, None, None

        if self.curr_model_dir != model_dir:
            self.model, self.labels = self.load_model_and_labels(model_dir)
            self.curr_model_dir = model_dir

        img = image.load_img(img_path, target_size=(self.input_width, self.input_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = self.model.predict(x)[0].tolist()
        if self.debug:
            print(preds)

        entropy = scipy.stats.entropy(preds)
        conf = {}
        best_label = self.labels[preds.index(max(preds))]
        for i in range(len(self.labels)):
            conf[self.labels[i]] = preds[i]

        if self.debug:
            print(best_label, entropy, conf)

        return best_label, entropy, conf

    ''' trains the object recognition model with the bottleneck features and saves the model and labels to files.
    	The bottleneck features are used to train the model faster.
    	
    	ATTENTION: This function is not implemented completely yet. This does not train a model correctly.

            Arguments:
                - model_dir: the directory to save the model and labels
                - img_dir: the directory with training samples (images)

            Returns:

    '''
    def train_with_bottleneck(self, model_dir, img_dir):
        start_time = time.time()
        if self.debug:
            print('Start training ...', img_dir)

        start_time = time.time()
        bottleneck_features, bottleneck_labels, labels = self.get_bottleneck_features(img_dir)
        bottleneck_labels_vector = to_categorical(bottleneck_labels)
        if self.debug:
            print('bottleneck features are collected: ', time.time() - start_time)

        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_features.shape[1:]))
        model.add(Dense(len(labels), activation='softmax'))  # final layer with softmax activation

        for l in model.layers:
            l.trainable = True
        # model.layers[-1].trainable = True

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if self.debug:
            print('setting up the training: ', time.time() - start_time)

        step_size_train = len(bottleneck_labels) // 50 # batch size is 50
        model.fit(bottleneck_features, bottleneck_labels_vector, steps_per_epoch=step_size_train, shuffle=True,
                  epochs=200)  # 200? # 80: around 2 minutes, 200: around 5 minutes, 100: current

        if self.debug:
            print('training is done: ', time.time() - start_time)

        # set current model and labels
        self.curr_model_dir = model_dir
        self.labels = labels
        self.model = model

        # save the trained model and labels
        self.save_model_and_labels(model_dir)

        if self.debug:
            print('saving the model to a file: ', time.time() - start_time)

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
            self.model, self.labels = self.load_model_and_labels(model_dir)
            self.curr_model_dir = model_dir

        img = image.load_img(img_path, target_size=(self.input_width, self.input_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feature = self.base_model.predict(x)

        preds = self.model.predict(feature)[0].tolist()
        if self.debug:
            print(preds)

        entropy = scipy.stats.entropy(preds)
        conf = {}
        best_label = self.labels[preds.index(max(preds))]
        for i in range(len(self.labels)):
            conf[self.labels[i]] = preds[i]

        if self.debug:
            print(best_label, entropy, conf)

        return best_label, entropy, conf
        
    def train(self, model_dir, img_dir):
        train_res = self.train_without_bottleneck(model_dir, img_dir)
        self.predict(model_dir, '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Remote/1.jpg') # warm up
        return train_res
        
    def predict(self, model_dir, img_path):
        return self.predict_without_bottleneck(model_dir, img_path)
#         return self.predict_with_bottleneck(model_dir, img_path)
    
    def reset(self, model_dir):
        try:
            rmtree(model_dir)
        except OSError as e:
            print("Reset error: %s : %s" % (model_dir, e.strerror))

if __name__ == '__main__':
    # test codes
    orec = ObjectRecognizer()
    orec.debug = True
    orec.train('model', '/Users/jonggihong/Downloads/tmpImages')
    best_label, _, _ = orec.predict('model', '/Users/jonggihong/Downloads/tmpImages/Remote/1.jpg')
    print(best_label)
    best_label, _, _ = orec.predict('model', '/Users/jonggihong/Downloads/tmpImages/Omega3/1.jpg')
    print(best_label)
    best_label, _, _ = orec.predict('model', '/Users/jonggihong/Downloads/tmpImages/Knife/1.jpg')
    print(best_label)


    # base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(orec.input_width, orec.input_height, 3))
    # bottleneck_features, bottleneck_labels, labels = orec.get_bottleneck_features('/Users/jonggihong/Downloads/tmpImages', base_model)


    # server test cases
    # orec.train('/home/jhong12/TOR-app-files/models/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83', '/home/jhong12/TOR-app-files/photo/TrainFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Spice')
#     orec.predict('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4', 
#     '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Remote/1.jpg')
#     orec.predict('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4', 
#     '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Knife/1.jpg')
#     orec.predict('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4', 
#     '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Omega3/1.jpg')



    # orec.train_without_bottleneck('/home/jhong12/TOR-app-files/models/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83',
    # '/home/jhong12/TOR-app-files/photo/TrainFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/Spice')
    # orec.predict_without_bottleneck('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4',
    # '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Remote/1.jpg')
    # orec.predict_without_bottleneck('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4',
    # '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Knife/1.jpg')
    # orec.predict_without_bottleneck('/home/jhong12/TOR-app-files/models/72F80764-EA2B-4B74-93B6-C4CA584551A4',
    # '/home/jhong12/TOR-app-files/photo/TrainFiles/72F80764-EA2B-4B74-93B6-C4CA584551A4/Spice/Omega3/1.jpg')
