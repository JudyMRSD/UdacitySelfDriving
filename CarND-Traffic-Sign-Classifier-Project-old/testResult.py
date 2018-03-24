import cv2
import glob
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from util import *

from network import *

# evaluate the model
with tf.Session() as sess:
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


    saver = tf.train.Saver()
    saver.restore(sess, "./result/model.ckpt")
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepareDataPipeline()

    print("X_test shape", X_test.shape)
    print("Y_test shape", y_test.shape)
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    # load and plot images
    # def testModel(numClass):
    numClass = 43
    tf.reset_default_graph()

    # load images
    path = './traffic-signs-data/googleImg/*.jpg'
    imageList = []
    for fileName in glob.glob(path):
        # print(fileName)
        img = cv2.imread(fileName, cv2.COLOR_BGR2RGB)
        # match image size from training data
        img = cv2.resize(img, (32, 32))
        imageList.append(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    # convert to numpy list
    X_test = np.asarray(imageList)
    # preprocess
    X_test = Y_channel_YUV(X_test)
    X_test = normalize(X_test)
    y_test = [14, 13, 34, 2, 25]

    saver.restore(sess, "./result/model.ckpt")
    print("X_test shape", X_test.shape)
    print("Y_test shape", y_test.shape)

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))






























