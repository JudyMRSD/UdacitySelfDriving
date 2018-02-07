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


def trainModel():
    # location to store tensorboard output
    log_dir_train = './log/train'
    log_dir_test = './log/test'

    EPOCHS = 2
    BATCH_SIZE = 128
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepareDataPipeline()
    numClass = len(np.unique(y_train))

    # uncomment to visualize data
    # visualizeData(X_train, y_train)

    # pipeline
    # x is a placeholder for a batch of input images.
    # y is a placeholder for a batch of output labels.
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, numClass)
    # Create a training pipeline that uses the model to classify MNIST data.
    rate = 0.001
    logits = LeNet(x, numClass)
    # Computes softmax cross entropy between logits and labels.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    # tf.reduce_mean(input_tensor, axis)
    # If axis has no entries, all dimensions are reduced,
    # and a tensor with a single element is returned.
    loss_operation = tf.reduce_mean(cross_entropy)
    # tensorboard
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    # minimize() takes care of both computing the gradients
    # and applying them to the variables
    training_operation = optimizer.minimize(loss_operation)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_operation)
    saver = tf.train.Saver()

    merged = tf.summary.merge_all()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    # train the model
    # Run the training data through the training pipeline to train the model.
    # Before each epoch, shuffle the training set.
    # After each epoch, measure the loss and accuracy of the validation set.
    # Save the model after training.
    # You do not need to modify this section.
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(log_dir_train, sess.graph)
        test_writer = tf.summary.FileWriter(log_dir_test)

        sess.run(tf.global_variables_initializer())

        num_examples = len(X_train)
        print("training")
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                #print("offset + BATCH_SIZE * i", offset + BATCH_SIZE * i)

                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                _, summary = sess.run([training_operation, merged], feed_dict={x: batch_x, y: batch_y})

                train_writer.add_summary(summary, offset + num_examples * i)
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        save_path = saver.save(sess, "./result/model.ckpt")
        train_writer.close()
        print("Model saved")

    # evaluate the model
    with tf.Session() as sess:
        saver.restore(sess, "./result/model.ckpt")
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# test a model on new images

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
        cv2.imshow('img',img)
        cv2.waitKey(0)
    # convert to numpy list
    X_test = np.asarray(imageList)
    # preprocess
    X_test = Y_channel_YUV(X_test)
    X_test = normalize(X_test)
    y_test = [14, 13, 34, 2, 25]
    print(X_test.shape)#(5, 32, 32, 1)




    #saver = tf.train.Saver()


    with tf.Session() as sess:
        # restore model
        # saver.restore(sess, "./result/model.ckpt")
        new_saver = tf.train.import_meta_graph('./result/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./result/'))

        # forward pass to make predictions
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        logits = LeNet(x, numClass)
        logits_prediction = sess.run(logits, feed_dict={x: X_test})
        predictions = np.argmax(logits_prediction, axis=1)

        print(predictions)
        match = (predictions==y_test)
        print (match)
        print(np.sum(match)/len(match))



        # output top 5 softmax probabilities
        new_saver = tf.train.import_meta_graph('./result/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./result/'))

        x = tf.placeholder(tf.float32, (None, 32, 32, 1))

        top_5 = tf.nn.top_k(logits, k=5)
        sess.run(top_5, feed_dict={x: X_train})

        for i in range (X_train.shape[0]):
            print("image",i, "top 5: ", top_5)



trainModel()
numClass = 43
#testModel(43)


































