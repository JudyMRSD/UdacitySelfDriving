import tensorflow as tf

# intuition: each neuron should focus on different things,
# thus the standard deviation for weights and biases should grow
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

# keep_prob = 1, no dropout
# final fc layer: logitsLayer = True
def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, keep_prob = 1, logitsLayer = False):
    """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        weights = tf.Variable(tf.truncated_normal(shape=(input_dim, output_dim), stddev=0.1))
        variable_summaries(weights)
        biases = tf.Variable(tf.zeros(output_dim))
        variable_summaries(biases)
        preactivate = tf.matmul(input_tensor, weights) + biases

        # only perform wx + b if it's the final fc layer
        if (logitsLayer == True):
            logits = preactivate
            return logits
        # if it's not a final fc layer, perform activation and dropout
        activation = act(preactivate, name = 'activation')
        if (keep_prob == 1):
            return activation
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(activation, keep_prob)

    return dropped



