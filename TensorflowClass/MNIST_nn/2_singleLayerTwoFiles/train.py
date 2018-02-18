import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from LeNet import LeNet
# LeNet here stands for a single layer network , not the actual lenet

def run_training(num_epoch, batch_size, lr):
    mainNN = LeNet()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    print("starts training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(num_epoch):
            # plt.imshow(mnist.train.image[0])
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed = {mainNN.x: batch_x, mainNN.labels: batch_y}
            loss, opt = sess.run([mainNN.loss, mainNN.opt], feed_dict=feed)
            if (ep % 10 == 0):
                accuracy = sess.run(mainNN.accuracy,
                                    feed_dict={mainNN.x: mnist.train.images, mainNN.labels: mnist.train.labels})
                print("loss= ", loss, "accuracy = ", accuracy)


def main():
    num_epoch = 1000
    batch_size = 128
    lr = 0.5
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    main()

