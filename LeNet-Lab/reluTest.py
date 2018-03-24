import tensorflow as tf


# relu:  f(x) = max(0,x)
test1 = tf.nn.relu(5)
test2 = tf.nn.relu(-1)

with tf.Session() as sess:
   print (sess.run(test1))
   print(sess.run(test2))
