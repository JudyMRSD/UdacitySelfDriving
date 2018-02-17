import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from LeNet import LeNet
from sklearn.utils import shuffle
from util import *

# LeNet here stands for a single layer network , not the actual lenet

def run_training(num_epoch, batch_size, learning_rate, model_save_dir):
    log_dir = './result'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # build LeNet
    lenet = LeNet()

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(lenet.loss)

    # test tensor board
    # tf.summary.scalar('loss', lenet.loss)
    # merged = tf.summary.merge_all()

    # loading data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)


    X_train = mnist.train.images
    X_train = np.reshape(X_train, [-1,28,28,1])
    y_train = mnist.train.labels
    num_examples = X_train.shape[0]

    print("starts training")

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # Create a saver.
        # The tf.train.Saver must be created after the variables that you want to restore (or save).
        # Additionally it must be created in the same graph as those variables.
        saver = tf.train.Saver(max_to_keep=10)


        # each epoch will shuffle the entire training data
        for ep in range(num_epoch):
            X_train, y_train =  shuffle(X_train, y_train)

            # train on each batch
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                feed = {lenet.x: batch_x, lenet.labels: batch_y}
                _, loss, summary = sess.run([train_step, lenet.loss, lenet.merged], feed_dict=feed)
                # print("summary", summary)
                train_writer.add_summary(summary, offset+num_examples*ep)

            # test on training data
            print("loss=", loss)

            accuracy = sess.run(lenet.accuracy, feed_dict=feed)

            print("accuracy = ", accuracy)
            # save model every other epoch
            if ep % 1 == 0:
                # Append the step number to the checkpoint name:
                saver.save(sess, model_save_dir+'/my-model', global_step=ep)

def test(model_save_dir):


    # load the graph structure from the ".meta" file into the current graph.
    tf.reset_default_graph()
    lenet = LeNet()
    # loading data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    X_test = mnist.test.images
    X_test = np.reshape(X_test, [-1, 28, 28, 1])
    y_test = mnist.test.labels
    # load the values of variables.
    # values only exist within a session
    # evaluate the model
    with tf.Session() as sess:
        var_list = tf.global_variables()

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, tf.train.latest_checkpoint('./model/lenet5/'))

        feed = {lenet.x: X_test, lenet.labels: y_test}
        test_accuracy = sess.run(lenet.accuracy, feed_dict=feed)
        print("Test Accuracy = {:.3f}".format(test_accuracy))




def main():
    num_epoch = 2
    batch_size = 128
    lr = 0.01
    model_save_dir = './model/lenet5'
    # run_training(num_epoch, batch_size, lr, model_save_dir)
    test(model_save_dir)

if __name__ == '__main__':
    main()

