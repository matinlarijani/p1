# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:41:53 2018

@author: Matin rohani larijani
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class LayerBuilder:
    def __init__(self):
        self.Input = None 
        self.Input_X = None
        self.Input_Y = None
        self.Input_depth = None
        self.Output_depth=None
        self.Filter_X_size = None
        self.Filter_Y_size = None
        self.Filter_stride = 1
        self.MaxPool_X_size = 2
        self.MaxPool_Y_size = 2
        self.MaxPool_strid = 2
        self.Name = None

    def build(self):
    # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [self.Filter_X_size, self.Filter_X_size,  self.Input_depth, self.Output_depth]
    
        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=self.Name+'_W')
        bias = tf.Variable(tf.truncated_normal([self.Output_depth]), name=self.Name+'_b')
    
        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(self.Input, weights, [1, self.Filter_stride, self.Filter_stride, 1], padding='SAME', name= self.Name)
    
        # add the bias
        out_layer += bias
    
        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)
    
        # now perform max pooling
        # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
        # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
        # applied to each channel
        ksize = [1, self.MaxPool_X_size,self.MaxPool_Y_size, 1]
        # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
        # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
        # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
        # to do strides of 2 in the x and y directions.
        strides = [1, self.MaxPool_strid, self.MaxPool_strid, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
        return out_layer





# Python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 50

#placeholder input
raw_image = tf.placeholder(tf.float32, [None, 784], name = 'raw_image')
image_shaped = tf.reshape(raw_image, [-1, 28, 28, 1], name = 'image_shaped')

#placeholder output
output = tf.placeholder(tf.float32, [None, 10], name ='output')

first_layer_filter = LayerBuilder();

first_layer_filter.Input = image_shaped
first_layer_filter.Input_X = 28
first_layer_filter.Input_Y = 28
first_layer_filter.Input_depth =1
first_layer_filter.Output_depth =32
first_layer_filter.Filter_X_size = 5
first_layer_filter.Filter_Y_size = 5
first_layer_filter.MaxPool_X_size =2
first_layer_filter.MaxPool_Y_size =2
first_layer_filter.Name = 'first_layer_'

layer1 = first_layer_filter.build()

second_layer_filter = LayerBuilder();

second_layer_filter.Input = layer1
second_layer_filter.Input_X = 28
second_layer_filter.Input_Y = 28
second_layer_filter.Input_depth =32
second_layer_filter.Output_depth =64
second_layer_filter.Filter_X_size = 5
second_layer_filter.Filter_Y_size = 5
second_layer_filter.MaxPool_X_size =2
second_layer_filter.MaxPool_Y_size =2
second_layer_filter.Name = 'second_layer_'

layer2 = second_layer_filter.build()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
# from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
# "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2, name='prediction')


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=output))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

# setup recording variables
# add a summary to store the accuracy
tf.summary.scalar('accuracy', accuracy)

# initiate a saver to save the model
saver = tf.train.Saver()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('PycharmProjects')
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={raw_image: batch_x, output: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={raw_image: mnist.test.images, output: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
        summary = sess.run(merged, feed_dict={raw_image: mnist.test.images, output: mnist.test.labels})
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={raw_image: mnist.test.images, output: mnist.test.labels}))
    print("\nSaving the model!")
    saver.save(sess, 'C:\\PyProjects\\nc\\number_classifier_model')
