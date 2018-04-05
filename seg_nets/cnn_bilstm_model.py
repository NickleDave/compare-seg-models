# class encapsulation of model adapted from
# https://danijar.com/structuring-your-tensorflow-models/
# and
# https://blog.metaflow.fr/
# tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3

from math import ceil

import tensorflow as tf

from .tf_utils import define_scope

def out_width(in_width, filter_width, stride):
    return ceil(float(in_width - filter_width + 1) / float(stride))


class CNNBiLSTM:
    def __init__(self,
                 n_syllables,
                 batch_size=11,
                 input_vec_size=513,
                 conv1_filters=32,
                 conv2_filters=64,
                 pool1_size=(1, 8),
                 pool1_strides=(1, 8),
                 pool2_size=(1, 8),
                 pool2_strides=(1, 8),
                 learning_rate=0.001,
                 max_to_keep=5
                 ):
        """hybrid convolutional neural net with bidirectional LSTM layer

        Arguments
        ---------
        spectrogram : tf.placeholder
            placeholder for training data.
            gets reshaped to (batch_size, spectrogram width, spectrogram height, 1 channel)
            spectrogram height is the same as "input vec size"
        num_hidden : int
            number of hidden layers in LSTMs
        seq_length : tf.placeholder
            holds sequence length
            equals time_steps * batch_size, where time_steps is defined by user as a constant
        n_syllables : int
            number of syllable types
            used as shape of output
        batch_size : int
            number of items in a batch.
            length of axis 0 of 3-d input array (spectrogram)
            default is 11.
        input_vec_size : int
            length of axis 3 of 3-d input array
            number of frequency bins in spectrogram
            default is 513
        """
        self.n_syllables = n_syllables
        self.batch_size = batch_size
        self.input_vec_size = input_vec_size
        self.conv1_filter = conv1_filters
        self.conv2_filter = conv2_filters
        self.pool1_size = pool1_size
        self.pool1_strides = pool1_strides
        self.pool2_size = pool2_size
        self.pool2_strides = pool2_strides
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_graph()
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)

        init = tf.global_variables_initializer()

    def build_graph(self):
        """build graph for 'inferring' labels of
        birdsong syllables from spectrogram
        """
        self.X = tf.placeholder("float",
                           [None,
                            None,
                            self.input_vec_size],
                           name="Xdata")
        self.Y = tf.placeholder("int32",
                           [None, None],
                           name="Ylabels")
        self.seq_length = tf.placeholder("int32",
                                         name="nSteps")

        conv1 = tf.layers.conv2d(inputs=tf.reshape(self.X,
                                                   [self.batch_size,
                                                    -1,
                                                    self.input_vec_size,
                                                    1]),
                                 filters=self.conv1_filters,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=self.pool1_size,
                                        strides=self.pool1_strides)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.conv2_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=self.pool2_size,
                                        strides=self.pool2_strides)

        # Determine number of hidden units in bidirectional LSTM:
        # uniquely determined by number of filters and frequency bins
        # in output shape of pool2
        freq_bins_after_pool1 = out_width(self.input_vec_size,
                                          self.pool1_size[1],
                                          self.pool1_strides[1])
        freq_bins_after_pool2 = out_width(freq_bins_after_pool1,
                                          self.pool2_size[1],
                                          self.pool2_strides[1])
        num_hidden = freq_bins_after_pool2 * self.conv2_filters

        # dynamic bi-directional LSTM
        lstm_f1 = tf.contrib.rnn.BasicLSTMCell(num_hidden,
                                               forget_bias=1.0,
                                               state_is_tuple=True,
                                               reuse=None)
        lstm_b1 = tf.contrib.rnn.BasicLSTMCell(num_hidden,
                                               forget_bias=1.0,
                                               state_is_tuple=True,
                                               reuse=None)
        outputs, _states = tf.nn.bidirectional_dynamic_rnn(lstm_f1,
                                                           lstm_b1,
                                                           tf.reshape(pool2,
                                                                      [self.batch_size,
                                                                       -1,
                                                                       num_hidden]
                                                                      ),
                                                           time_major=False,
                                                           dtype=tf.float32,
                                                           sequence_length=self.seq_length)

        # projection on the number of syllables creates logits time_steps
        with tf.name_scope('Projection'):
            W_f = tf.Variable(tf.random_normal([num_hidden, self.n_syllables]))
            W_b = tf.Variable(tf.random_normal([num_hidden, self.n_syllables]))
            bias = tf.Variable(tf.random_normal([self.n_syllables]))

        expr1 = tf.unstack(outputs[0],
                           axis=0,
                           num=self.batch_size)
        expr2 = tf.unstack(outputs[1],
                           axis=0,
                           num=self.batch_size)
        logits = tf.concat([tf.matmul(ex1, W_f) + bias + tf.matmul(ex2, W_b)
                            for ex1, ex2 in zip(expr1, expr2)],
                           0)
        self.logits = logits
        self.outputs = outputs


    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits

    @define_scope
    def optimize(self):
        xentropy_layer = xentropy(logits=self.logits,
                                  labels=tf.concat(tf.unstack(self.Y,
                                                              axis=0,
                                                              num=self.batch_size),
                                                   0),
                                  name='xentropy')
        self.cost = tf.reduce_mean(xentropy_layer, name='cost')
        tf.summary.scalar("loss", self.cost)
        merged_summary_op = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        return optimizer.minimize(self.cost, global_step=self.global_step)

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent-ep_' + str(episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:

        json.dump(self.config, f)
