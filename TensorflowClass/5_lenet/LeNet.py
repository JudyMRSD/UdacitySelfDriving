# Lenet
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
# VAE
# https://github.com/allenovo/conditional_vae/blob/master/vae.py
# MNIST
# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
# Tensorboard:
# http://blog.csdn.net/sinat_33761963/article/details/62433234

# group layers
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
#
import tensorflow as tf
from util import *
from tensorflow.contrib.layers import flatten


class LeNet():
    def __init__(self, ):
        numInputs = 28*28
        numClass = 10
        learning_rate = 0.5
        with tf.name_scope('LeNet'):
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, [None, numInputs], name='input')
                self.x = tf.reshape(self.x, shape=[-1, 28, 28, 1])
                self.labels = tf.placeholder(tf.float32, [None], name='output')
                one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=numClass)

            #with tf.name_scope('input_reshape'):
            #    image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            #    tf.summary.image('input', image_shaped_input, 10)
            # name scope : layer name
            # 5 lenet layers goes here
            # conv1, conv2, conv3, fc1, fc2
            #
            # conv1 layer
            # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x6.
            # filter: (5, 5, 1, 6)
            conv1 = conv_layer(filter_side=5, input_tensor=self.x, out_channels=6, layer_name='conv1')

            # Flatten
            # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
            f0 = flatten(conv1)

            # final fc layer
            self.logits = fc_layer(f0, numClass, 'fc_layer', logitsLayer = True)
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=self.logits)
                self.loss = tf.reduce_mean(cross_entropy)
            #with tf.name_scope('train'):
            #    train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            # calculate accuracy
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            # merge the summaries
            self.merged = tf.summary.merge_all()
            #print("self.merged", self.merged)












