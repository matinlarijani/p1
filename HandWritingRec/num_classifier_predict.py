# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:45:37 2018

@author: matin rohani larijani

this script uses the modle trained in num_classifier_cnn-main to predict sample images 
"""


import tensorflow as tf
import numpy as np
import os,cv2
import sys

from os import listdir
from os.path import isfile, join

filename = ''
image_size=28
num_channels=1
images = []

mypath = 'C:\\numbers'
for f in listdir(mypath):
    filename = join(mypath, f)
    if isfile(filename):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
                



images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
number_of_images = len(images)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(number_of_images, image_size,image_size,num_channels)


saver = tf.train.import_meta_graph('C:\\PyProjects\\nc\\number_classifier_model.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('C:\\PyProjects\\nc\\'))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()
    
    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph..get_tensor_by_name("raw_image:0")
    
    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("raw_image:0") 
    y_true = graph.get_tensor_by_name("output:0") 
    y_test_images = np.zeros((1, 2)) 
    
    
    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print(result)