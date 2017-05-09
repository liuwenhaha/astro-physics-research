import argparse
import os
import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# parameters for network
import time

batch_size = 100
# hidden unit for pixel: 3-12
# hidden unit for psf: 91-100
hidden1_units = 7
FLAGS = None

# placeholder for data
coord_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
psf_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2034))



def inference(coords, hidden1_units):
    # Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([2, hidden1_units], stddev=1.0/math.sqrt(2034.0)))
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(coords, weights)+biases)
    # TODO: Try to add Hidden2
    # Output
    with tf.name_scope('linear_output'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, 2034],
                                                  stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([2034]), name='biases')
        # TODO: Try linear and relu
        psf_value = tf.matmul(hidden1, weights) + biases
    return psf_value


def loss(pixel_values, pixel_labels):
    return tf.reduce_mean(tf.pow(pixel_labels - pixel_values, 2), name='mean_square_error')


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


def placeholder_inputs(batch_size):
    coord_placeholder = tf.placeholder(tf.float32, shape=(batch_size,2))
    psf_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2034))
    return coord_placeholder, psf_labels_placeholder


def fill_feed_dict(data_set, coord_pl, psf_labels_pl):
    coord_feed, psf_label_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        coord_pl: coord_feed,
        psf_labels_pl: psf_label_feed
    }
    return feed_dict


class DataSet:

    def __init__(self, coord, psf_labels):
        """Construct a DataSet.
        """
        assert coord.shape[0] == psf_labels.shape[0], (
            'coord.shape: %s psf_labels.shape: %s' % (coord.shape, psf_labels.shape))
        print('coord.shape: %s psf_labels.shape: %s' % (coord.shape, psf_labels.shape))
        self._num_examples = coord.shape[0]
        self._coord = coord
        self._psf_labels = psf_labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def coord(self):
        return self._coord

    @property
    def psf_labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._coord = self.coord[perm0]
            self._psf_labels = self.psf_labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            coord_rest_part = self._coord[start:self._num_examples]
            psf_labels_rest_part = self._psf_labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._coord = self.coord[perm]
                self._psf_labels = self.psf_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            coord_new_part = self._coord[start:end]
            psf_labels_new_part = self._psf_labels[start:end]
            return (np.concatenate((coord_rest_part, coord_new_part), axis=0),
                    np.concatenate((psf_labels_rest_part, psf_labels_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._coord[start:end], self._psf_labels[start:end]


# TODO: check what to do with this method
def do_eval(sess, coord_placeholder, psf_labels_placeholder, data_set):
    cur_loss = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, coord_placeholder, psf_labels_placeholder)
        # TODO: Accumulate the loss
        # loss += evaluation()
    print('Num examples:{} Loss{}'.format(num_examples, cur_loss))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


def execute(hidden1=128):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    # TODO: Modify this to use passed in para and optimize it
    parser.add_argument(
        '--hidden1',
        type=int,
        default=hidden1,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    # TODO: Adjustable para
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)