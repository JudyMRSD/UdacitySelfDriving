import pickle
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

# load pickled data

def loadData():
    # TODO: Fill this in based on where you saved the training and testing data
    data_folder = "./traffic-signs-data/"
    training_file = data_folder + "train.p"
    validation_file = data_folder + "valid.p"
    testing_file = data_folder + "test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_train_coord = train['coords'];
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    assert (len(X_train) == len(y_train))
    assert (len(X_valid) == len(y_valid))
    assert (len(X_test) == len(y_test))

    '''
    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_valid)))
    print("Test Set:       {} samples".format(len(X_test)))

    '''

    return X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test


### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

def exploreDataSet(X_train, y_train):
    X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()
    print("number of keys in train.p as dicitonary : ", len(train))
    print("keys: ", train.keys())# dict_keys(['features', 'coords', 'sizes', 'labels'])

    # (num examples, width, height, channels)
    print("X_train.shape", train['features'].shape) # (34799, 32, 32, 3)
    print("bounding box coord", train['coords'].shape) # (34799, 4)
    # a list containing tuples, (width, height)
    print("sizes of original image (w,h)", train['sizes'].shape)  # (34799, 2), num example x 2

    # Number of training examples
    # len(y_train) will give the same result
    n_train = len(X_train)  # 34799

    # Number of validation examples
    n_validation = len(y_valid)  # 12630

    # Number of testing examples.
    n_test = len(y_test)

    # X_train[0] the first training example
    image_shape = X_train[0].shape
    # X_train[0].shape returns a tuple
    print("image_h = ", X_train.shape[1])
    print("image_w = ", X_train.shape[2])
    print("image_channel = ", X_train.shape[3])

    # TODO: How many unique classes/labels there are in the dataset.
    # set(y_train): unique elements
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
#uncomment to explore dataset
#exploreDataSet(X_train, y_train)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
def visualize(X_train, y_train):
    # bounding box
    imageIndex = 100
    print("single image shape", X_train[imageIndex].shape)

    fig, ax = plt.subplots(1)
    ax.imshow(X_train[imageIndex], cmap="gray")
    x1, y1, x2, y2 = X_train['coords'][imageIndex]

    # Create a Rectangle patch  x y width height
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    # classes is the ordered unique classes

    # indices:  indices of input array that result in the unique array classes
    # index of examples that is the first occurence of a sign,
    # the first 40 will be the first 40 classes

    # counts: the number of times each unique sign appears

    classes, indices, counts = np.unique(y_train, return_index=True, return_counts=True)
    print("list of classes", classes)  # [0,...42]
    num_classes = len(classes)
    print("number of classes", num_classes)  # 43

    # plotting the count of each sign
    # historgram bins arranged by classes
    print("historgram bins arranged by classes ")
    plt.hist(y_train, classes)
    plt.show()

    # unique_images are the first occurence of traffic signs in the data set
    unique_images = X_train[indices]
    print("num unique images : ", len(unique_images))

    fig = plt.figure(figsize=(20, 5))
    for i in range(num_classes):
        ax = fig.add_subplot(5, 9, i + 1, xticks=[], yticks=[])
        ax.imshow(unique_images[i])
    plt.show()

    # uncomment to visualize dataset
    # visualize(X_train, y_train)


def visualizeAugmented(X_train, y_train, X_coord):
    # plotting the count of each sign



    # bounding box
    imageIndex = 100
    print("single image shape", X_train[imageIndex].shape)

    fig, ax = plt.subplots(1)
    ax.imshow(np.squeeze(X_train[imageIndex]), cmap='gray')
    x1, y1, x2, y2 = X_coord[imageIndex]

    # Create a Rectangle patch  x y width height
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    classes, indices, counts = np.unique(y_train, return_index=True, return_counts=True)
    print("list of classes", classes)  # [0,...42]
    num_classes = len(classes)
    print("number of classes", num_classes)  # 43

    # historgram bins arranged by classes
    print("historgram bins arranged by classes ")
    plt.hist(y_train, classes)
    plt.show()

    # unique_images are the first occurence of traffic signs in the data set
    unique_images = X_train[indices]
    print("num unique images : ", len(unique_images))

    fig = plt.figure(figsize=(20, 5))
    for i in range(num_classes):
        ax = fig.add_subplot(5, 9, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(unique_images[i]), cmap='gray')
    plt.show()


def testSplitChannel():
    # test YUV split
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
    img = cv2.imread("test.png")
    print("b channel")
    b1, g, r = cv2.split(img)
    plt.imshow(b1)
    plt.show()
    b2 = img[:, :, 0]
    plt.imshow(b2)
    plt.show()

    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv_img)
    print("yuv")
    plt.imshow(yuv_img)
    plt.show()

    print("y channel")
    print("y channel size", y.shape)
    plt.show()
    plt.imshow(y)
    plt.show()

    print("y channel")
    y2 = yuv_img[:, :, 0]
    plt.show()
    print("y2 shape", y2.shape)

    # gray
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray")
    print(gray_image.shape)
    plt.imshow(gray_image)
    plt.show()

# testSplitChannel()

# test print normalized image
def testNormalized():
    X_train_coord , X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()
    print("X_train[0].shape",X_train[0].shape)

    def normalize(X):
        # Normalized Data  to 0 to 1  normalized = (x-min(x))/(max(x)-min(x))
        X = (X - np.amin(X))/(np.amax(X) - np.amin(X))
        #X = (X-128)/128

        return X


    X_train = normalize(X_train)

    plt.imshow(X_train[0])
    plt.show()

def testGray():
    X_train_coord , X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()
    # test change to gray scale
    img = X_train[0,:,:,:]
    plt.imshow(img)
    plt.show()
    plt.imshow(X_train[0])
    plt.show()
    # yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #y , u, v = cv2.split(yuv_img)
    #X[i] = y
    #if (i<5):
    #    plt.imshow(X[i])
    #    plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    plt.imshow(gray, cmap='gray')
    plt.show()
    gray = np.expand_dims(gray, axis=2)
    print("gray.shape",gray.shape)


    X_train[0] = gray

    plt.imshow(X_train[0])
    plt.show()


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to YUV, etc.
### Feel free to use as many code cells as needed.

# zero mean for data X, X can be X_train, X_valid or X_test
# X is Y channel of YUV image
# Feature scaling is used to bring all values into the range [0,1].
# This is also called unity-based normalization.
# X' = (X-X_min)/(X_max - X_min)

# convert color image to YUV then extract Y channel
# Opencv reference
# https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gga4e0972be5de079fed4e3a10e24ef5ef0adc0f8a1354c98d1701caad4b384e0d18

# input:  shape = (numExample x 32 x 32 x 3)
# output:  shape = (numExample x 32 x 32 x 1)
def Y_channel_YUV(X):
    threeChannelShape = X.shape
    # shape is tuple, not mutable
    singleChannelShape = threeChannelShape[0:3] + (1,)
    # set to single channel
    X_singleChannel = np.zeros(singleChannelShape)

    for i in range(0, len(X)):
        img = X[i]
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv_img)
        y = np.expand_dims(y, axis=2)
        X_singleChannel[i] = y

    # plt.imshow(X[0], cmap='gray')
    # plt.show()

    return X_singleChannel


def normalize(X):
    # Normalized Data  to 0 to 1  normalized = (x-min(x))/(max(x)-min(x))
    X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))  # if input X (None, 32, 32 , 3), output X (None, 32, 32 , 3)
    # X = (X-128)/128

    return X


def oneHotLabel(y_train, y_valid, y_test):
    numClass = len(np.unique(y_train))
    y_train = tf.one_hot(indices=tf.cast(y_train, tf.int32), depth=numClass)
    y_valid = tf.one_hot(indices=tf.cast(y_valid, tf.int32), depth=numClass)
    y_test = tf.one_hot(indices=tf.cast(y_test, tf.int32), depth=numClass)
    return y_train, y_valid, y_test


def preprocess(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # Y channel
    X_train = Y_channel_YUV(X_train)

    X_valid = Y_channel_YUV(X_valid)
    X_test = Y_channel_YUV(X_test)
    # normalize gray images
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)
    # one hot encoding labels
    # y_train, y_valid, y_test = oneHotLabel(y_train, y_valid, y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def prepareDataPipeline():
    # Step 1: Import data

    X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()

    # Step 2: Use data agumentation to make more training data
    # X_train, y_train = dataAugmentation(X_train, y_train)

    # Step 3: Data processing for tarin, validation, and test dataset
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Step 4: visualize preprocessed data
    # visualizeAugmented(X_train, y_train, X_train_coord)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


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
    keep_prob = 1

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
    # print("conv1",conv1.get_shape())

    # Activation
    # relu:  f(x) = max(0,x)
    conv1 = tf.nn.relu(conv1)
    # dropout
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # max pooling reduce by 1/2
    # Input = 32x32x6. Output = 16x16x6.
    # new_height = 32/2 = 16
    # new_width = 32/2 = 16
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
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


def testTF():
    x = tf.placeholder(tf.float32, (100, 32, 32, 1))
    y = tf.placeholder(tf.int32, (100))
    y_hot = tf.placeholder(tf.int32, (100, 43))
    one_hot_y = tf.one_hot(y, 43)

    print(y_hot.get_shape())  # (100, 43)
    print("tf.one_hot=", one_hot_y.get_shape())  # (100, 43)

# testTF()

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

def trainModel():
    # location to store tensorboard output
    log_dir_train = './log/train'
    log_dir_test = './log/test'

    EPOCHS = 20
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

        saver.save(sess, './lenet')
        train_writer.close()
        print("Model saved")

    # evaluate the model
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


trainModel()







