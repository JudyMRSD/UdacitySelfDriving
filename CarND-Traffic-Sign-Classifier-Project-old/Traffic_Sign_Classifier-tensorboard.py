
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Jin Zhu
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# useful references for data preprocessing:
# https://github.com/udacity/aind2-cnn
# 
# Data augmentation:
# https://github.com/udacity/aind2-cnn/blob/master/cifar10-augmentation/cifar10_augmentation.ipynb
# 
# Data preparation:
# https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258
# 
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# In[57]:

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


# ---
# ## Step 0: Load The Data

# In[58]:

#load pickled data

def loadData():
    # TODO: Fill this in based on where you saved the training and testing data
    data_folder = "./traffic-signs-data/"
    training_file = data_folder + "train.p"
    validation_file= data_folder + "valid.p"
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
    
    
    return X_train_coord , X_train, y_train, X_valid, y_valid, X_test, y_test


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[59]:

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


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[60]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
def visualize(X_train, y_train):
    # bounding box 
    imageIndex = 100
    print("single image shape",X_train[imageIndex].shape)

    fig,ax = plt.subplots(1)
    ax.imshow(X_train[imageIndex], cmap="gray")
    x1, y1, x2 , y2 =  train['coords'][imageIndex]
    
    # Create a Rectangle patch  x y width height
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    # classes is the ordered unique classes 

    # indices:  indices of input array that result in the unique array classes
    # index of examples that is the first occurence of a sign, 
    # the first 40 will be the first 40 classes

    # counts: the number of times each unique sign appears

    classes, indices, counts = np.unique(y_train, return_index=True,return_counts=True)
    print ("list of classes",classes) # [0,...42]
    num_classes = len(classes)
    print ("number of classes",num_classes)  # 43

    # plotting the count of each sign
    # historgram bins arranged by classes 
    print ("historgram bins arranged by classes ")
    plt.hist(y_train, classes)
    plt.show()


    # unique_images are the first occurence of traffic signs in the data set 
    unique_images = X_train[indices]
    print ("num unique images : ",len(unique_images))

    fig = plt.figure(figsize=(20,5))
    for i in range (num_classes):
        ax = fig.add_subplot(5, 9, i+1,  xticks=[], yticks=[])    
        ax.imshow(unique_images[i])
    plt.show()

#uncomment to visualize dataset    
# visualize(X_train, y_train)


# In[61]:

def visualizeAugmented(X_train, y_train, X_coord):
    # plotting the count of each sign
    
    
    
    # bounding box 
    imageIndex = 100
    print("single image shape",X_train[imageIndex].shape)

    fig,ax = plt.subplots(1)
    ax.imshow(np.squeeze(X_train[imageIndex]),cmap='gray')
    x1, y1, x2 , y2 =  X_coord[imageIndex]
    
    # Create a Rectangle patch  x y width height
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    
    classes, indices, counts = np.unique(y_train, return_index=True,return_counts=True)
    print ("list of classes",classes) # [0,...42]
    num_classes = len(classes)
    print ("number of classes",num_classes)  # 43

    
    # historgram bins arranged by classes 
    print ("historgram bins arranged by classes ")
    plt.hist(y_train, classes)
    plt.show()

    
    # unique_images are the first occurence of traffic signs in the data set 
    unique_images = X_train[indices]
    print ("num unique images : ",len(unique_images))

    fig = plt.figure(figsize=(20,5))
    for i in range (num_classes):
        ax = fig.add_subplot(5, 9, i+1,  xticks=[], yticks=[])   
        ax.imshow(np.squeeze(unique_images[i]),cmap='gray')
    plt.show()
    
    
    

#uncomment to visualize dataset    
#visualize(X_train, y_train)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (data augmentation, normalization, grayscale, etc.)

# In[62]:

def testSplitChannel():
    # test YUV split
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
    img = cv2.imread("test.png")
    print("b channel")
    b1,g,r = cv2.split(img)
    plt.imshow(b1)
    plt.show()
    b2 = img[:,:,0]
    plt.imshow(b2)
    plt.show()
    
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y , u, v = cv2.split(yuv_img)
    print("yuv")
    plt.imshow(yuv_img)
    plt.show()
    
    print("y channel")
    print ("y channel size",y.shape )
    plt.show()
    plt.imshow(y)
    plt.show()
    
    print("y channel")
    y2 = yuv_img[:,:,0]
    plt.show()
    print("y2 shape",y2.shape)
    
    # gray
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray")
    print(gray_image.shape)
    plt.imshow(gray_image)
    plt.show()
     
    
#testSplitChannel()


# In[63]:

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


# In[64]:

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



# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.
# 
# 
# 
# Jin:
# according to the paper : http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
# build an augmented dataset:
# perterb in position [-2, 2] pixels,  scale [.9, 1.1] ratio, rotation [-15, +15] degrees , affine transformation, brightness
# 
# 
# 
# 

# In[65]:

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
    singleChannelShape = threeChannelShape[0:3]+(1,)
    # set to single channel 
    X_singleChannel = np.zeros(singleChannelShape)
    
    
    for i in range (0, len(X)):
        img = X[i]
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y , u, v = cv2.split(yuv_img)
        y = np.expand_dims(y, axis=2)
        X_singleChannel[i] = y
         
    #plt.imshow(X[0], cmap='gray')
    #plt.show()
    
    return X_singleChannel
    
def normalize(X): 
    # Normalized Data  to 0 to 1  normalized = (x-min(x))/(max(x)-min(x))
    X = (X - np.amin(X))/(np.amax(X) - np.amin(X))   # if input X (None, 32, 32 , 3), output X (None, 32, 32 , 3)
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


# In[66]:

# helper functions for data augmentation to generate more training data
# data augmentation
def dataAugmentation():
    return 0


# ### Part 2.1 : Prepare Dataset for traing, validation, test
# 
# Steps: 
# 
# Step 1: Import data
# 
# Step 2: Use data agumentation to make more training data 
# 
# Step 3: Data processing for tarin, validation, and test dataset 
#     
# 
# 

# In[67]:

def prepareDataPipeline():
    
    # Step 1: Import data

    X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test= loadData()

    #Step 2: Use data agumentation to make more training data 
    # X_train, y_train = dataAugmentation(X_train, y_train)

    # Step 3: Data processing for tarin, validation, and test dataset 
    X_train, y_train, X_valid, y_valid, X_test, y_test  = preprocess(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Step 4: visualize preprocessed data
    # visualizeAugmented(X_train, y_train, X_train_coord)
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test 
    
#prepareDataPipeline()


# ### Part 2.2 :  Model Architecture
# 
#  
# Modifications on LeNet:
# 
# 1. One more convolutional layer ： higher level feature, better accuracy
# 
#    first layer convolutional layer is lower level features, higher the layer -> higher level featuer
# 
# 2. Add dropout to avoid overfitting
# 
# 
# 
# 

# In[68]:

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

    # tf.Variable for trainable variables
    # such as weights (W) and biases (B) for your model.

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # -----------------------------------------------------------
    # filter : (height, width, input_depth, output_depth) = (F, F, 1, 6)
    # F = 5:
    #   Caculation of filter size F for valid padding:
    #   out_height = ceil(float(in_height) / float(strides[1]))
    #    out_width  = ceil(float(in_width) / float(strides[2]))

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
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, numClass), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(numClass))
    logits = tf.matmul(fc2, fc3_W)+fc3_b

    return logits


# ### Train, Validate and Test the Model

# In[69]:

def testTF():
    x = tf.placeholder(tf.float32, (100, 32, 32, 1))
    y = tf.placeholder(tf.int32, (100))
    y_hot = tf.placeholder(tf.int32, (100, 43))
    one_hot_y = tf.one_hot(y, 43)
    
    
    print(y_hot.get_shape())# (100, 43)
    print("tf.one_hot=",one_hot_y.get_shape()) # (100, 43)
    
#testTF()


# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[70]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

def trainModel():
    EPOCHS = 10
    BATCH_SIZE = 128
    log_dir_train = './log'
    X_train, y_train, X_valid, y_valid, X_test, y_test  = prepareDataPipeline()
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
    tf.summary.scalar('loss', loss_operation)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    # minimize() takes care of both computing the gradients
    # and applying them to the variables
    training_operation = optimizer.minimize(loss_operation)
    
    
    
    

        
        
    # model evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tf.summary.scalar('accuracy', accuracy_operation)
    merged = tf.summary.merge_all()

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
        train_writer = tf.summary.FileWriter(log_dir_train, sess.graph)

        
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("training")
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                #sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})
                _, summary = sess.run([training_operation, merged], feed_dict={x:batch_x, y:batch_y})
                train_writer.add_summary(summary, offset+ BATCH_SIZE*i)
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


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[ ]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


# ### Analyze Performance

# In[ ]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

