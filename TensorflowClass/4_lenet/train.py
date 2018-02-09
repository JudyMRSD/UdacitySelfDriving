import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from LeNet import LeNet
from sklearn.utils import shuffle

# LeNet here stands for a single layer network , not the actual lenet

def run_training(num_epoch, batch_size, learning_rate):
    log_dir = './result'
    # build LeNet
    lenet = LeNet()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(lenet.loss)

    # loading data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


    X_train = mnist.train.images
    y_train = mnist.train.labels
    num_examples = X_train.shape[0]



    print("starts training")
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # each epoch will shuffle the entire training data
        for ep in range(num_epoch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            X_train, y_train = shuffle(X_train, y_train)
            # train on each batch
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                feed = {lenet.x: batch_x, lenet.labels: batch_y}
                _, loss = sess.run([train_step, lenet.loss], feed_dict=feed)
            # test on training data
            accuracy = sess.run(lenet.accuracy, feed_dict=feed)
            print("loss= ", loss, "accuracy = ", accuracy)


def main():
    num_epoch = 1
    batch_size = 128
    lr = 0.5
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    main()

