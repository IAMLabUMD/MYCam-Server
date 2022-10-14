# Server for the MYCam App

<a href="https://jonggi.github.io"><img src="https://img.shields.io/badge/contact-Jonggi Hong-blue.svg?style=flat" alt="Contact"/></a>
<a href="LICENSE.md"><img src="https://img.shields.io/badge/license-TBD-red.svg?style=flat" alt="License: TBD"/></a>
<img src="https://img.shields.io/badge/platform-Linux-green"/> 
<img src="https://img.shields.io/badge/language-Python 3.6-lightblue"/>

This page explains [codes](https://github.com/IAMLabUMD/MYCam-Server) on the server side communicating with [the mobile MYCam app](https://github.com/IAMLabUMD/MYCam-Mobile). The main role of the server is as follows:
1. It trains an Inception V3 model with photos from the user.
2. It recognizes objects with the model
3. It calculates the attributes of a photo or a set of photos. 

You can access the codes here. https://github.com/IAMLabUMD/MYCam-Server


## Requirements
In order to run the MYCam app, you will need to meet the following requirements:
```
- Python 3.6
- Ubuntu 16.04
- Tensorflow 2.0 
- CUDA 8.0
```

## Getting started
To build and run the MYCam app, please follow these steps,
1. Set up the environment. See instructions [here](https://www.tensorflow.org/install/pip) to find out how to set up tensorflow in Ubuntu 16.04.
2. Run `TOR_HTTP_Server_v3.py` with the following command.
```
    python3 TOR_HTTP_Server_v3.py
```

## Scripts
These are brief descriptions of the classes and functions. For more details, please read the comments in the files.

### HTTP Server 
`TOR_HTTP_Server_v3.py`

This is a simple HTTP server that calls functions in classes for using an InceptionV3 model and descriptors. You can run the server with the following script.
```python
httpd = HTTPServer(('127.0.0.1', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
```

### InceptionV3 model
`ObjectRecognizerV2.py`
* load_model_and_labels(self, model_dir): load a model and label files in the 'model_dir' directory
* save_model_and_labels(self, save_dir, org_dir): move a model in 'org_dir' to 'save_dir'
* get_bottleneck_features(self, img_dir): saves bottleneck features of the images in 'img_dir' directory
* train(self, model_dir, img_dir): train a pre-trained InceptionV3 model with images in 'img_dir' and save the model in the 'model_dir' directory
* predict(self, model_dir, img_path): recognize an object in the image at 'img_path' using the model in the 'model_dir' directory

### Calculating attributes of a photo(s)
`DescriptorGenerator.py`
* getImageDescriptor(self, img_path): calculate attributes of a single image at 'img_path'
* getSetDescriptor(self, arinfo_path): calculate the attributes of a set of images using the information from ARKit at 'arinfo_path' including the positions of a camera, side of object, etc.
---
`HandSegmentation.py`

It has a class to find pixels of a hand in an image. The hand segmentation can be used as below.
```python
model ='TOR_hand_all+TOR_feedback_fcn8_10k_16_1e-5_450x450'
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

image = cv2.imread('/Users/jonggihong/Downloads/tmp_2.jpg', cv2.IMREAD_COLOR)
if width is not None and height is not None:
    new_shape = (int(width), int(height))
    image = cv2.resize(image, new_shape, cv2.INTER_CUBIC)

# localize an object from the input image
image, pred = segmentation.do(image)
hand_area = np.sum(pred)
```
---

`ObjectDetector.py`

This is a YOLOv3 object detector. See the example of usage below.

* def detect(self, image_path): detect an object in an image at 'image_path'

```python
od = ObjectDetector()
od.detect('/home/jhong12/TOR-app-files/photo/TempFiles/CA238C3A-BDE9-4A7F-8CCA-76956A9ABD83/tmp_2.jpg')
```

### Others
`CHI2017_retrain.py`
: A script for transfer learning used for a study in Hernisa's CHI 2017 paper.

---

`GORTest.py`
: Measuring the accuracy of an InceptionV3 model trained with photos of 15 objects used in Kyungjun's ASSETS 2019 study. 

---

`ObjectRecognizer.py`
: Old script for using an InceptionV3 model which is replaced with `ObjectRecognizerV2.py` 

---

`StudyHelper.py`
: Helper functions for the user study with the MYCam app

---

`retrain.py`
: Script for transfer learning from a tensorflow website.

---


## Publications
Under review

## Contact
Jonggi Hong <jhong12@umd.edu>

Hernisa Kacorri <hernisa@umd.edu>
