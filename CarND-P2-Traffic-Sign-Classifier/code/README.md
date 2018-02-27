# **Traffic Sign Classifier Pipeline** 
-----
# train.py:
## main():
step0: define hyper parameters, learning rate = 0.01  <br/>
step1: call prepareDataPipeline to get preprocessed data.  <br/>
step2: call run-training on training dataset, and printout accuracy on validation dataset  <br/>


## run-training: 
Use LeNet, inputs are 32x32x3 RGB images , gradient descent is done using Adam optimizer <br/>
During traning, use validation set every 10 episode to check the accracy.<br/>

## test: 
load saved model and test on test data set  <br/>
Please ignore this function for now, since it's not used during training. <br/>


# LeNet.py:

LeNet class defines the graph for a modified LeNet structure: <br/>
conv1 with relu: output (?, 32, 32, 6)    max pool:  (?, 16, 16, 6) <br/>
conv2 with relu: output (?, 16, 16, 16)   max pool:  (?, 8, 8, 16) <br/> 
conv3 with relu: output  (?, 8, 8, 16)    max pool:  (?, 4, 4, 16) <br/>
fc0 is the conv3 output flattened to (?, 256) <br/>
fc1   (?, 120)  <br/>
fc2   (?, 84)  <br/>
logits (?, num class)  <br/>
Cross entropy loss was used here.  <br/>

# tfUtil.py
Helper functions to build the graph in LeNet.py that's friendly to display graph and write summary for variables on tensorboard.  <br/>
conv-layer function creates a convolution layer with activation, maxpool and option for dropout. <br/>
fc-layer function creates an fc layer with activation if it's a hidden layer, without activaation if it's a logits layer.  <br/>

# dataUtil.py:
prepareDataPipeline loads the data and normalize the images. <br/>
normalizeAll normalizes images in test, train, and validation sets using X_data / 255. - 0.5   <br/>

Other functions that I implemented but commented out:  <br/>
dataAugmentation :  create data agumentaion with balanced number of examples per class. <br/>
visualize: this function displays example images for each class in the dataset , and plot a histogram for the distribution of data for each class.<br/>
preprocess: take Y channel from YUV as indicated in the recommended paper. The single channel images are later normalized  <br/>
testTF: test one hot encoding in tensorflow.   <br/>

# testTF.py
Please ignore this file, used for testing tensorflow functions. 


