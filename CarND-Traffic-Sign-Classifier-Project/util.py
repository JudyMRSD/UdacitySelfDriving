import cv2
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten



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


    return X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

def exploreDataSet(X_train, y_train):
    X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()
    print("number of keys in train.p as dicitonary : ", len(X_train))
    print("keys: ", X_train.keys())# dict_keys(['features', 'coords', 'sizes', 'labels'])

    # (num examples, width, height, channels)
    print("X_train.shape", X_train['features'].shape) # (34799, 32, 32, 3)
    print("bounding box coord", X_train['coords'].shape) # (34799, 4)
    # a list containing tuples, (width, height)
    print("sizes of original image (w,h)", X_train['sizes'].shape)  # (34799, 2), num example x 2

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

# test changing image to grayscale
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


def testTF():
    x = tf.placeholder(tf.float32, (100, 32, 32, 1))
    y = tf.placeholder(tf.int32, (100))
    y_hot = tf.placeholder(tf.int32, (100, 43))
    one_hot_y = tf.one_hot(y, 43)

    print(y_hot.get_shape())  # (100, 43)
    print("tf.one_hot=", one_hot_y.get_shape())  # (100, 43)
