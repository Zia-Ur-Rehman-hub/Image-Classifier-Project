# Image-Classifier-Project
The aim of this project is to classify the image of a flower from Oxford Flower Set Dataset and identify which class this flower image belong to.

#                                                    Oxford FLowers Image CLassifier Project
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset from Oxford of 102 flower categories, you can see a few examples below.


#                                                             Project Steps
# Part 1:
Load the image dataset and create a pipeline.
Build and Train an image classifier on this dataset.
Use your trained model to perform inference on flower images.
We'll lead you through each part which you'll implement in Python.
# Part 2:
When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. 

#                                                           Folders and files of Project:
# train.py:
This file contains the script which will load image dataset,create pipeline and will build and train image classifier.
# predict.py
This file contains script which will load the saved model and will perform a prediction of image and will classify class of flower.This file will be run using command line arguments.
# test_model.h5:
This is our trained model saved in h5 file format
# test Images folder:
This folder contains sample images of some flowers which will be further used to verify the prediction of our model.
# label_map.json:
This file will map  the predicted image class (which is no ) to actual names of flower class.

# Oxford Flowers Image Classifier Project.ipynb
This file contains the complete source code which consists of parts of training model and then predicting flower class.



