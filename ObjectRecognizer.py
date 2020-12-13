"""
    A teachable object recognizer class with MobileNetV2 and transfer learning

    Author: Jonggi Hong
    Date: 12/13/2020
"""

import time
import numpy as np
import sys
import os
import scipy.stats
from pathlib import Path

from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input

class ObjectRecognizer:
    def __init__(self):
        self.debug = False
        self.input_width = 224
        self.input_height = 224
        self.curr_model_dir = 'no model is loaded.'

    """ loads the classification model and labels
        
        Arguments:
            - model_dir: the directory with the model and label files
        
        Returns:
            - model: keras model instance from the model file
            - labels: list of labels
    """
    def loadModelAndLabels(self, model_dir):
        model_path = os.path.join(model_dir, 'model.h5')
        model = load_model(model_path)

        label_path = os.path.join(model_dir, 'labels.txt')
        labels = []
        with open(label_path) as openfileobject:
            for line in openfileobject:
                labels.append(line.strip())

        return model, labels

    """ trains the object recognition model and saves the model and labels to files

            Arguments:
                - model_dir: the directory to save the model and labels
                - img_dir: the directory with training samples (images)

            Returns:
            
    """
    def train(self, model_dir, img_dir):
        if self.debug:
            print("Start training ...", img_dir)

        start_time = time.time()
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
        train_generator = train_datagen.flow_from_directory(img_dir,
                                                            target_size=(224, 224),
                                                            color_mode='rgb',
                                                            batch_size=30,
                                                            class_mode='categorical',
                                                            shuffle=True)

        labels = (train_generator.class_indices)
        labels2 = dict((v, k) for k, v in labels.items())
        if self.debug:
            print('training data are collected: ', time.time() - start_time)

        # imports the mobilenet model and discards the last 1000 neuron layer.
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.input_width, self.input_height, 3))

        if self.debug:
            print('loading the base model (', base_model.name, '): ', time.time() - start_time)

        x = base_model.output
        x = Flatten()(x)
        preds = Dense(len(labels2), activation='softmax')(x)  # final layer with softmax activation

        model = Model(inputs=base_model.input, outputs=preds)
        model.layers[-1].trainable = True

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if self.debug:
            print('setting up the training: ', time.time() - start_time)

        step_size_train = train_generator.n // train_generator.batch_size
        model.fit(train_generator, steps_per_epoch=step_size_train, epochs=100)  # 200? # 80: around 2 minutes, 200: around 5 minutes

        if self.debug:
            print('training is done: ', time.time() - start_time)

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.h5')
        label_path = os.path.join(model_dir, 'labels.txt')

        # save the model to file
        model.save(model_path)

        # save the labels to file
        f = open(label_path, 'w')
        for i in range(len(labels2)):
            f.write(labels2[i] + '\n')
        f.close()

        # set current model and labels
        self.curr_model_dir = model_dir
        self.labels = labels2
        self.model = model

        if self.debug:
            print('saving the model to a file: ', time.time() - start_time)

    """ predicts the object in an image

            Arguments:
                - model_dir: the directory with the model and labels
                - img_path: the target image

            Returns:
                - best_label: the label with the highest confidence score
                - entropy: entropy of the confidence scores
                - conf: a dictionary with confidence scores of all labels (label, confidence score)
    """
    def predict(self, model_dir, img_path):
        if self.curr_model_dir != model_dir:
            self.model, self.labels = self.loadModelAndLabels(model_dir)
            self.curr_model_dir = model_dir

        img = image.load_img(img_path, target_size=(self.input_width, self.input_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = self.model.predict(x)[0]
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

if __name__ == "__main__":
    orec = ObjectRecognizer()
    orec.debug = True
    # orec.train('model', '/Users/jonggihong/Downloads/tmpImages')
    orec.test('model', '/Users/jonggihong/Downloads/tmpImages/Knife/1.jpg')