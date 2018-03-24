import cv2

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


'''
conv1 (?, 32, 32, 6)
conv1 (?, 16, 16, 6)
conv2 (?, 16, 16, 16)
conv2 (?, 8, 8, 16)
conv3 (?, 8, 8, 16)
conv3 (?, 4, 4, 16)
fc0 (?, 256)
fc1 (?, 120)
fc2 (?, 84)
logits (?, 43)

'''


def LeNet(X_train, numClass):
    # Architecture:
    # ------------------------
    # # Arguments used for tf.truncated_normal,
    # randomly defines variables for the weights and biases for each layer

    # truncated normal:
    # truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,seed=None, name=None)
    # generated values follow a normal distribution with specified mean and standard deviation,
    # except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.

    mu = 0
    sigma = 0.1
    keep_prob = 0.5

    # tf.Variable for trainable variables
    # such as weights (W) and biases (B) for your model.

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x6.
    # -----------------------------------------------------------
    # filter : (height, width, input_depth, output_depth) = (F, F, 1, 6)
    # F = 5:
    #   Caculation of filter size F for valid padding:
    #   out_height = ceil(float(in_height) / float(strides[1]))
    #   out_width  = ceil(float(in_width) / float(strides[2]))
    #   out_height = 32/1
    #   out_width  = 32/1
    # conv1_W is the filter dimension
    # 6 filters, each is 5x5x1   width = 5, height = 5, input channel = 1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    # conv2d(input, filter, strides, padding)
    # input tensor : [batch, in_height, in_width, in_channels]
    # filter: [filter_height, filter_width, in_channels, out_channels]
    conv1 = tf.nn.conv2d(X_train, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    print("after conv2d,  conv1",conv1.get_shape())

    # Activation
    # relu:  f(x) = max(0,x)
    conv1 = tf.nn.relu(conv1)
    

    # max pooling reduce by 1/2
    # Input = 32x32x6. Output = 16x16x6.
    # new_height = 32/2 = 16
    # new_width = 32/2 = 16
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # dropout
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # print("conv1",conv1.get_shape())
    # Layer 2: Convolutional. Input = 16x16x6. Output = 16x16x16.
    # -----------------------------------------------------------
    #  out_height = 16/1=16
    #  out_width = 16/1=16
    # filter : (height, width, input_depth, output_depth)

    # 16 filters, each is 5x5x6   width = 5, height = 5, input channel = 6
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    # print("conv2",conv2.get_shape())
    # Activation
    conv2 = tf.nn.relu(conv2)
    # max pooling  input 16x16x16. output 8x8x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print("conv2",conv2.get_shape())
    # add one more layer than Lenet, to learn more higher level features
    # (first layers learns low level features)
    # dropout
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Layer 3: Convolutional. Input = 8x8x16. Output = 8x8x16.
    # -----------------------------------------------------------
    #  out_height = 16/1=16
    #  out_width = 16/1=16
    # filter : (height, width, input_depth, output_depth)

    # 16 filters, each is 5x5x6   width = 5, height = 5, input channel =16
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 16), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(16))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    # print("conv3",conv3.get_shape())
    # Activation
    conv3 = tf.nn.relu(conv3)
    # max pooling  input 8x8x16. output 4x4x16
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print("conv3",conv3.get_shape())
    # dropout
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Flatten
    # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
    # Input = 4x4x16. Output = 256.
    # tf.contrib.layers.flatten,
    #   Flattens the input while maintaining the batch_size.
    #   input  [batchsize ,...]
    #   The flatten function flattens a Tensor into two dimensions: (batches, length).
    #   The batch size remains unaltered,
    #   so all of the other dimensions of the input Tensor are flattened into the second dimension of the output Tensor.
    fc0 = flatten(conv3)
    # print("fc0",fc0.get_shape())
    # Now that the Tensor is 2D, it's ready to be used in fully connected layers.
    # Layer 3: Fully Connected. This should have 120 outputs.
    # ------------------------------------------------
    # matrix multiplication dimension  axb    x   bxc   = axc
    # num neurons 120
    #  1x256    x   256x120   =   1x120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(256, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))

    # Multiplies matrix a by matrix b, producing a * b.
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    #  activation
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # print("fc1",fc1.get_shape())
    #  Layer 4: Fully Connected. This should have 84 outputs.
    # ------------------------------------------------
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    # Multiplies matrix a by matrix b, producing a * b.
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # activation
    fc2 = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # print("fc2",fc2.get_shape())
    #  Layer 5: Fully Connected. This should have 10 outputs.
    # ------------------------------------------------
    # logits:  For Tensorflow: It's a name that it is thought to imply
    # that this Tensor is the quantity that is being mapped to probabilities by the Softmax
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, numClass), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(numClass))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    # print("logits",logits.get_shape())
    return logits

