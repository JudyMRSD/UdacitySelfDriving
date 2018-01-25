import tensorflow as tf


#case 2
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("case 2")
    print(sess.run(op2))# output 1x3x3x1

# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
# another example:
k = tf.constant([
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
], dtype=tf.float32, name='k')

i = tf.constant([
    [4, 3, 1, 0],
    [2, 1, 0, 1],
    [1, 2, 4, 1],
    [3, 1, 0, 2]
], dtype=tf.float32, name='i')

kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
print(kernel)
image  = tf.reshape(i, [1, 4, 4, 1], name='image')
print(image)
res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID"))
# VALID means no padding
with tf.Session() as sess:
   print (sess.run(res))






