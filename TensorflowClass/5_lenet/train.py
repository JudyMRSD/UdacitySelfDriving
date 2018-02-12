import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from LeNet import LeNet
from sklearn.utils import shuffle
from util import *

# LeNet here stands for a single layer network , not the actual lenet

def run_training(num_epoch, batch_size, learning_rate):
    log_dir = './result'
    # build LeNet
    lenet = LeNet()

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(lenet.loss)

    # test tensor board
    # tf.summary.scalar('loss', lenet.loss)
    # merged = tf.summary.merge_all()

    # loading data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


    X_train = mnist.train.images
    X_train = np.reshape(X_train, [-1,28,28,1])
    y_train = mnist.train.labels
    num_examples = X_train.shape[0]

    print("starts training")

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # each epoch will shuffle the entire training data
        for ep in range(num_epoch):
            X_train, y_train =  shuffle(X_train, y_train)
            for i in range (0, 5):
                showImg(X_train[i])
                print("y_train[i]", y_train[i])

            # train on each batch
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                feed = {lenet.x: batch_x, lenet.labels: batch_y}


                _, summary = sess.run([train_step, lenet.merged], feed_dict=feed)
                # print("summary", summary)
                train_writer.add_summary(summary, offset+num_examples*ep)
            # test on training data
            accuracy = sess.run(lenet.accuracy, feed_dict=feed)

            print("accuracy = ", accuracy)


def main():
    num_epoch = 1
    batch_size = 128
    lr = 0.5
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    main()

