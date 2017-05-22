import argparse
import os
import tensorflow as tf
import math
import numpy as np
import psf_interpolation_utils as utils
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# parameters for network
import time

# batch_size = 100
# hidden unit for pixel: 3-12
# hidden unit for psf: 91-100
# hidden1_units = 7
# FLAGS = None

# placeholder for data
# coord_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
# psf_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2304))



def inference(coords, hidden1_units, hidden2_units):
    # Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([2, hidden1_units], stddev=1.0/math.sqrt(2.0)))
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(coords, weights)+biases)

    # TODO: Try to add Hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0/math.sqrt(hidden1_units)))
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights)+biases)
    # Output
    with tf.name_scope('linear_output'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, 2304],
                                                  stddev=1.0/math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([2304]), name='biases')
        # TODO: Try linear and relu
        psf_value = tf.matmul(hidden2, weights) + biases

    # # Output
    # with tf.name_scope('linear_output'):
    #     weights = tf.Variable(tf.truncated_normal([hidden1_units, 2304],
    #                                               stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
    #     biases = tf.Variable(tf.zeros([2304]), name='biases')
    #     # TODO: Try linear and relu
    #     psf_value = tf.matmul(hidden1, weights) + biases

    return psf_value


def loss(pixel_values, pixel_labels):
    return tf.reduce_mean(tf.squared_difference(pixel_labels, pixel_values), name='mean_square_error')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    # TODO: Try different optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op


def evaluation(pixel_values, pixel_labels):
    # Here use basic mse
    # TODO: Find other better indicator
    return tf.reduce_mean(tf.pow(pixel_labels - pixel_values, 2))


def placeholder_inputs(batch_size):
    coord_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
    psf_labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2304))
    return coord_placeholder, psf_labels_placeholder


def fill_feed_dict(data_set, coord_pl, psf_labels_pl, FLAGS):
    coord_feed, psf_label_feed = data_set.next_batch(FLAGS['batch_size'])
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
        return self._psf_labels

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
def do_eval(sess, eval_correct, coord_placeholder, psf_labels_placeholder, data_set, FLAGS):
    """Runs one evaluation against the full epoch of data.
      Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
          input_data.read_data_sets().
      """
    # And run one epoch of eval.
    the_loss = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS['batch_size']
    num_examples = steps_per_epoch * FLAGS['batch_size']
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   coord_placeholder,
                                   psf_labels_placeholder,
                                   FLAGS)
        the_loss += sess.run(eval_correct, feed_dict=feed_dict)
    mean_loss = float(the_loss) / num_examples
    print('  Num examples: %d  Total loss: %0.09f  Mean loss @ 1: %0.09f' %
          (num_examples, the_loss, mean_loss))


def run_training(data_sets, FLAGS):
  """Train MNIST for a number of steps."""
  # TODO: Refer to tensorboard example to add multiple network support
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    coord_placeholder, psf_labels_placeholder = placeholder_inputs(
        FLAGS['batch_size'])

    # Build a Graph that computes predictions from the inference model.
    psf_pred = inference(coord_placeholder, FLAGS['hidden1'], FLAGS['hidden2'])

    # Add to the Graph the Ops for loss calculation.
    the_loss = loss(psf_pred, psf_labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(the_loss, FLAGS['learning_rate'])

    # Add the Op to compare the psf_pred to the labels during evaluation.
    eval_correct = evaluation(psf_pred, psf_labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS['log_dir'], sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS['max_steps']):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets['train'],
                                 coord_placeholder,
                                 psf_labels_placeholder,
                                 FLAGS)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, the_loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.5f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS['max_steps']:
        checkpoint_file = os.path.join(FLAGS['log_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        # TODO: Check do_eval
        do_eval(sess,
                eval_correct,
                coord_placeholder,
                psf_labels_placeholder,
                data_sets['train'],
                FLAGS)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                coord_placeholder,
                psf_labels_placeholder,
                data_sets['validate'],
                FLAGS)
        # TODO: Add test set if necessary
        # Evaluate against the test set.
        # print('Test Data Eval:')
        # do_eval(sess,
        #         eval_correct,
        #         images_placeholder,
        #         labels_placeholder,
        #         data_sets.test)



def execute(learning_rate=0.01, max_steps=2000, hidden1=128, hidden2=32, batch_size=100,
            log_dir='assets/log/w2m0m0_831555/lr0.01_ms2000_h1.128_h2.32_bs100',
            datasets=None):
    FLAGS = {}
    FLAGS['learning_rate'] = learning_rate
    FLAGS['max_steps'] = max_steps
    FLAGS['hidden1'] = hidden1
    FLAGS['hidden2'] = hidden2
    FLAGS['batch_size'] = batch_size
    FLAGS['log_dir'] = log_dir

    if tf.gfile.Exists(FLAGS['log_dir']):
        tf.gfile.DeleteRecursively(FLAGS['log_dir'])
    tf.gfile.MakeDirs(FLAGS['log_dir'])

    run_training(datasets, FLAGS)


def tf_psfwise_interpolation(self, learning_rate=0.01, hidden1=36, hidden2=144, max_steps=4000, batch_size=100):
    '''
    train neural network define in tf_psfwise_interpolation.py
    save the trained model to log_dir
    with cache support
    :return:
    '''

    # psf_data -> [{'chip_no': 1,
    #               'chip_data': [[x,y,RA,Dec,psf_numpy_48_48], ...],
    #               'chip_train_data': <view_of_chip_data>,
    #               'chip_validate_data': <view_of_chip_data>,},
    #              ...]

    # TODO: prepare datasets
    data_sets = {}
    for tag in ('train', 'validate'):
        coord = []
        psf_labels = []
        chip_data_name = 'chip_{}_data'.format(tag)
        for chip_psf_data in self.psf_data:
            coord += [data[2:4] for data in chip_psf_data[chip_data_name]]
            psf_labels += [data[4].ravel() for data in chip_psf_data[chip_data_name]]
        coord = np.array(coord)
        psf_labels = np.array(psf_labels)
        data_sets[tag] = DataSet(coord, psf_labels)
    # hidden unit for pixel: 3-12
    # hidden unit for psf: 91-100
    execute(learning_rate=learning_rate, max_steps=max_steps, hidden1=hidden1,
            hidden2=hidden2, batch_size=batch_size,
            log_dir='assets/log/{}_{}/l2_lr{}_ms{}_h1.{}_h2.{}_bs{}'
            # log_dir='assets/log/{}_{}/l1_lr{}_ms{}_h1.{}_bs{}'
            .format(self.region, self.exp_num, learning_rate, max_steps,
                   hidden1, hidden2, batch_size),
            datasets=data_sets)


def predict(self, coord, fits_info,
            learning_rate=0.01, max_steps=2000, hidden1=36, hidden2=144, batch_size=100):
    network_model_dir = 'assets/log/{}_{}/l2_lr{}_ms{}_h1.{}_h2.{}_bs{}/model.ckpt-{}'.format(self.region, self.exp_num, learning_rate, max_steps, hidden1, hidden2, batch_size, max_steps-1)
    with tf.Graph().as_default():
        num_coord = len(coord)
        coord_placeholder, psf_labels_placeholder = placeholder_inputs(num_coord)
        psf_pred = inference(coord_placeholder, hidden1, hidden2)
        new_saver = tf.train.Saver()
        sess = tf.Session()
        new_saver.restore(sess, network_model_dir)
        feed_dict = {
            coord_placeholder: coord
        }
        psf_predictions = sess.run(psf_pred, feed_dict=feed_dict)
        result_dir = 'assets/predictions/{}_{}/tf_psfwise/l2_lr{}_ms{}_h1.{}_h2.{}_bs{}/'.format(self.region, self.exp_num, learning_rate, max_steps, hidden1, hidden2, batch_size)
        utils.write_predictions(result_dir, psf_predictions, fits_info, method='tf_psfwise')

    #     psf_predictions = sess.run([psf_pred], feed_dict={})
    #     _, loss_value = sess.run([train_op, the_loss],
    #                              feed_dict=feed_dict)
    #     result_dir = 'assets/predictions/tf_psfwise/'
    #     utils.write_predictions(result_dir, psf_predictions, fits_info)
    # # TODO: Make dir more flexible
    # new_saver = tf.train.import_meta_graph(network_log_dir+'')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    # all_vars = tf.get_collection('vars')
    # for v in all_vars:
    #     v_ = sess.run(v)
    #     print(v_)
    #     batch_x =
    #     predictions = sess.run(y_hat, feed_dict={x: batch_x})
