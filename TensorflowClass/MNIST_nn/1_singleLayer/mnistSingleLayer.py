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
        learning_rate = 0.5
        with tf.variable_scope('simpleNN'):
           self.x= tf.placeholder(tf.float32, [None, numInputs], name = 'input')
           self.labels =  tf.placeholder(tf.float32, [None], name = 'output')
           one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=numClass)
           # one fullly connected layer
           self.W_fc0 = tf.Variable(tf.truncated_normal(shape = (numInputs, numClass), mean = 0, stddev = 1))
           self.b_fc0 = tf.Variable(tf.zeros(numClass))
           # [batch_size, num_classes]
           self.logits = tf.matmul(self.x, self.W_fc0) + self.b_fc0

           cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=self.logits)
           self.loss = tf.reduce_mean(cross_entropy)
           self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

           # calculate accuracy
           self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_labels,1))
           self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

def main():
    numEpochs = 1000
    mainNN = simpleNN()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    print(mnist.train.images[0].shape)
    print(mnist.train.labels[0])


    print ("starts training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for ep in range(numEpochs):
            # plt.imshow(mnist.train.image[0])
            batch_x, batch_y = mnist.train.next_batch(100)
            feed = {mainNN.x: batch_x, mainNN.labels: batch_y}
            loss, opt = sess.run([mainNN.loss, mainNN.opt], feed_dict = feed)
            if (ep%10 ==0):
                print("loss= ",loss)
                accuracy = sess.run(mainNN.accuracy,
                                    feed_dict={mainNN.x: mnist.train.images, mainNN.labels: mnist.train.labels})
                print(accuracy)


if __name__ == '__main__':
    main()

















