# Lenet
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
# VAE
# https://github.com/allenovo/conditional_vae/blob/master/vae.py
# MNIST
# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
# Tensorboard:
# group layers
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


class simpleNN():
    def __init__(self):
        numInputs = 28*28
        numClass = 10
        learning_rate = 0.0001
        with tf.variable_scope('simpleNN'):
           self.x= tf.placeholder(tf.float32, [None, numInputs], name = 'input')
           # one hot encoded labels
           # self.labels  [batch_size, num_classes]
           self.labels =  tf.placeholder(tf.float32, [None], name = 'output')
           # mnist dataset already one hot encoded the labels
           #one_hot_labels = tf.one_hot(self.labels, numClass)
           # one fullly connected layer
           self.W_fc0 = tf.Variable(tf.truncated_normal(shape = (numInputs, numClass), mean = 0, stddev = 1))
           self.b_fc0 = tf.Variable(tf.zeros(numClass))
           # [batch_size, num_classes]
           self.logits = tf.matmul(self.x, self.W_fc0) + self.b_fc0
           # define loss and optimizer
           # we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b),
           # because this more numerically stable function internally computes the softmax activation.

           # y = tf.nn.softmax(tf.matmul(x, W) + b)
           # y_ = tf.placeholder(tf.float32, [None, 10])
           # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
           cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits)
           self.loss = tf.reduce_mean(cross_entropy)
           self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            # model evaluation
           #self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
           #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
def main():
    numEpochs = 10
    mainNN = simpleNN()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images[0].shape)
    print(mnist.train.labels[0])

    '''
    print ("starts training")
    with tf.Session() as sess:
        sess.run(tf.global_variable_initializer())


        #plt.imshow(mnist.train.image[0])
        batch_x, batch_y = mnist.train.next_batch(100)
        feed = {mainNN.x: batch_x, mainNN.labels: batch_y}
        for ep in range(numEpochs):
            sess.run(mainNN.opt, feed_dict = feed)
            accuracy = sess.run(mainNN.accuracy, feed_dict=feed)
            print(accuracy)


    '''


if __name__ == '__main__':
    main()

















