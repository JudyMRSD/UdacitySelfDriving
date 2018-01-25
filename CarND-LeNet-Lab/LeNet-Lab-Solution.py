from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def loadData():
    # load data
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert (len(X_train) == len(y_train))
    assert (len(X_validation) == len(y_validation))
    assert (len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    return X_train, y_train, X_validation, y_validation, X_test, y_test


# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.

# However, the LeNet architecture only accepts 32x32xC images,
# where C is the number of color channels.
# In order to reformat the MNIST data into a shape that LeNet will accept,
# we pad the data with two rows of zeros on the top and bottom,
# and two columns of zeros on the left and right (28+2+2 = 32).

# pad image with 0s
# Image Shape: (28, 28, 1)

def preprocess():
    # 1. load data
    X_train, y_train, X_validation, y_validation, X_test, y_test = loadData()

    # 2. padding
    # print("Image Shape before padding: {}".format(X_train.shape)) # X_train: shape (55000, 28, 28, 1)
    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    # print("Image Shape after padding : {}".format(X_train.shape)) # X_train: shape (55000, 32, 32, 1)

    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    #("Updated Image Shape: {}".format(X_train[0].shape))

    # 3. Shuffle
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def visualizeData(X_train, y_train):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    plt.figure(figsize= (1,1))
    plt.imshow(image, cmap="gray")
    plt.show()
    print("y_train[index]",y_train[index])

# LeNet:
# input:
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.
# Since MNIST images are grayscale, C is 1 in this case.
#
# output:
# return the result of the 2nd fully connected layer

def LeNet(X_train):
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

    # tf.Variable for trainable variables
    # such as weights (W) and biases (B) for your model.

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # -----------------------------------------------------------
    # filter : (height, width, input_depth, output_depth) = (F, F, 1, 6)
    # F = 5:
    #   Caculation of filter size F for valid padding:
    #   out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    #   out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    #   out_height = ceil(float(32 - 5 + 1) / float(1)) = 28
    #   out_width = ceil(float(32 - 5 + 1) / float(1)) = 28
    conv1_W = tf.Variable(tf.truncated_normal(shape = (5,5,1,6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    # conv2d(input, filter, strides, padding)
    # input tensor : [batch, in_height, in_width, in_channels]
    # filter: [filter_height, filter_width, in_channels, out_channels]
    conv1 = tf.nn.conv2d(X_train, conv1_W, strides=[1,1,1,1], padding ='VALID')+conv1_b

    # Activation
    # relu:  f(x) = max(0,x)
    conv1 = tf.nn.relu(conv1)

    # max pooling
    # Input = 28x28x6. Output = 14x14x6.
    # new_height = ceil(float(28 - 2 + 1) / float(2)) = ceil(13.5) = 14
    # new_width = ceil(float(28 - 2 + 1) / float(2)) = ceil(13.5) = 14
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    # -----------------------------------------------------------
    #  out_height = ceil(float(14 - 5 + 1) / float(1)) = 28
    #  out_width = ceil(float(14 - 5 + 1) / float(1)) = 28
    # filter : (height, width, input_depth, output_depth)
    conv2_W = tf.Variable(tf.truncated_normal(shape = (5,5,6,16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides = [1,1,1,1], padding='VALID')+conv2_b
    # Activation
    conv2 = tf.nn.relu(conv2)
    # max pooling  input 10x10x16. output 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding='VALID')

    # Flatten
    # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
    # Input = 5x5x16. Output = 400.
    # tf.contrib.layers.flatten,
    #   Flattens the input while maintaining the batch_size.
    #   input  [batchsize ,...]
    #   The flatten function flattens a Tensor into two dimensions: (batches, length).
    #   The batch size remains unaltered,
    #   so all of the other dimensions of the input Tensor are flattened into the second dimension of the output Tensor.
    fc0 = flatten(conv2)

    # Now that the Tensor is 2D, it's ready to be used in fully connected layers.
    # Layer 3: Fully Connected. This should have 120 outputs.
    # ------------------------------------------------
    # matrix multiplication dimension  axb    x   bxc   = axc
    #  1x400    x   400x120   =   1x120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    # Multiplies matrix a by matrix b, producing a * b.
    fc1 = tf.matmul(fc0, fc1_W)+fc1_b

    #  activation
    fc1 = tf.nn.relu(fc1)

    #  Layer 4: Fully Connected. This should have 84 outputs.
    # ------------------------------------------------
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    # Multiplies matrix a by matrix b, producing a * b.
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # activation
    fc2 = tf.nn.relu(fc2)

    #  Layer 5: Fully Connected. This should have 10 outputs.
    # ------------------------------------------------
    # logits:  For Tensorflow: It's a name that it is thought to imply
    # that this Tensor is the quantity that is being mapped to probabilities by the Softmax
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W)+fc3_b

    return logits



def main():
    EPOCHS = 10
    BATCH_SIZE = 128


    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess()

    # uncomment to visualize data
    # visualizeData(X_train, y_train)

    # pipeline
    # x is a placeholder for a batch of input images.
    # y is a placeholder for a batch of output labels.
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)
    # Create a training pipeline that uses the model to classify MNIST data.
    rate = 0.001
    logits = LeNet(x)
    # Computes softmax cross entropy between logits and labels.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    # tf.reduce_mean(input_tensor, axis)
    # If axis has no entries, all dimensions are reduced,
    # and a tensor with a single element is returned.
    loss_operation = tf.reduce_mean(cross_entropy)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    # minimize() takes care of both computing the gradients
    # and applying them to the variables
    training_operation = optimizer.minimize(loss_operation)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy/num_examples

    # train the model
    # Run the training data through the training pipeline to train the model.
    # Before each epoch, shuffle the training set.
    # After each epoch, measure the loss and accuracy of the validation set.
    # Save the model after training.
    # You do not need to modify this section.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("training")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})

            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

    saver.save(sess, './lenet')
    print("Model saved")


    # evaluate the model
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

if __name__== "__main__":
  main()

