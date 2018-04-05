# Encoder-Decoder Temporal Convolutional Network
# adapted from [1]
# https://github.com/colincsl/TemporalConvolutionalNetworks
#
# [1]René, Colin Lea Michael D. Flynn, and Vidal Austin Reiter Gregory D. Hager.
# "Temporal convolutional networks for action segmentation and detection." (2017).


import tensorflow as tf

from .tf_utils import define_scope


class EncoderDecoderTCN:
    """Encoder Decoder Temporal Convolutional Network

    """
    def __init__(self,
                 num_freq_bins,
                 num_time_steps,
                 conv1_num_filters=64,
                 conv2_num_filters=96,
                 max_to_keep=5):
        self.num_freq_bins = num_freq_bins
        self.num_time_steps = num_time_steps
        self.conv1_num_filters = conv1_num_filters
        self.conv2_num_filters = conv2_num_filters
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder()
            self.build_graph()
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_graph(self):

        with tf.name_scope('encoder'):
            tf.nn.conv2d(input=self.input,
                         filter=(self.num_time_steps,
                                  self.num_freq_bins,
                                  self.conv1_num_filters),
                         strides=[1,1,1,1],
                         padding='same',
                         name='encoder_conv1')
