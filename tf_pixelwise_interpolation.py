import tensorflow as tf
import math

# parameters for network
batch_size = 100
# hidden unit for pixel: 3-12
# hidden unit for psf: 91-100
hidden1_units = 7


# placeholder for data
coord_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
pixel_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))


def inference(coords, hidden1_units):
    # Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([2, hidden1_units], stddev=1.0/math.sqrt(2.0)))
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(coords, weights)+biases)
    # Output
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, 1],
                                                  stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([1]), name='biases')
        pixel_values = tf.matmul(hidden1, weights) + biases
    return pixel_values


def loss(pixel_values, pixel_labels):
    return tf.reduce_mean(tf.pow(pixel_labels-pixel_values, 2), name='mean_square_error')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    # TODO: Try different optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op


def evaluation(pixel_values, pixel_labels):
    # Here use basic mse
    # TODO: Find other better indicator
    return tf.reduce_mean(tf.pow(pixel_labels - pixel_values, 2))
