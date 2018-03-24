# reference:
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
class LeNet:
    def __init__(self):
        # conv 1
        self.W_conv1 = tf.Variable