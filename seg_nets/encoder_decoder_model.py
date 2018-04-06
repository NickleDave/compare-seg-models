# Encoder-Decoder Network
# adapted in part from [1]
# https://github.com/colincsl/TemporalConvolutionalNetworks
#
# [1]René, Colin Lea Michael D. Flynn, and Vidal Austin Reiter Gregory D. Hager.
# "Temporal convolutional networks for action segmentation and detection." (2017).

import numpy as np
import tensorflow as tf

from .tf_utils import define_scope, spatial_dropout


def encoder_layer(input,
                  num_filters,
                  filter_shape,
                  padding='same',
                  random_seed=42,
                  use_spatial_dropout=True,
                  dropout_rate=0.3,
                  pool_size=2,
                  name=None):
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(filter_shape) + \
                        [n_input_channels, num_filters]
        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        conv = tf.nn.conv2d(input=input,
                            filter=filter_shape,
                            strides=[1, 1, 1, 1],
                            padding=padding,
                            name='_conv')
        conv = tf.nn.bias_add(conv, biases,
                              name='_bias')
        if use_spatial_dropout:
            sd2d = spatial_dropout(conv1, dropout_rate, random_seed)
            conv = tf.nn.relu(features=sd2d, name='_relu')
        else:
            conv = tf.nn.relu(features=conv, name='_relu')

        pool = tf.layers.max_pooling2d(inputs=conv,
                                       pool_size=[pool_size, 2],
                                       strides=2,
                                       name='_pool')
        return pool


def decoder_layer(input,
                  num_filters,
                  filter_shape,
                  height_factor=2,
                  width_factor=2,
                  padding='same',
                  random_seed=42,
                  use_spatial_dropout=True,
                  dropout_rate=0.3,
                  name=None):
    with tf.variable_scope(name):
        # adapted from Keras backend,
        # https://github.com/keras-team/keras/blob/
        # e2a10a5e6e156a45e946c4d08db7133f997c1f9a/keras/backend/tensorflow_backend.py#L1912
        # which is method used by Keras upsampling2d:
        # https://github.com/keras-team/keras/blob/
        # 8ed57c168f171de7420e9a96f9e305b8236757df/keras/layers/convolutional.py#L1552
        original_shape = input.get_shape().as_list()
        new_shape = tf.shape(input)[1:3]
        new_shape *= tf.constant(np.array([height_factor,
                                           width_factor]).astype('int32'))
        input = tf.image.resize_nearest_neighbor(input, new_shape)
        input.set_shape((None, original_shape[1] * height_factor
            if original_shape[1] is not None else None,
                             original_shape[2] * width_factor
                             if original_shape[2] is not None else None, None))

        input_shape = input.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(filter_shape) + \
                        [n_input_channels, num_filters]
        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        conv = tf.nn.conv2d(input=input,
                            filter=filter_shape,
                            strides=[1, 1, 1, 1],
                            padding=padding,
                            name='_conv')
        conv = tf.nn.bias_add(conv, biases,
                              name='_bias')
        if use_spatial_dropout:
            sd2d = spatial_dropout(conv1, dropout_rate, random_seed)
            conv = tf.nn.relu(features=sd2d, name='_relu')
        else:
            conv = tf.nn.relu(features=conv, name='_relu')
        return conv


def time_distributed(input, layer):
    """

    :param input:
    :param layer:
    :return:
    """
    input_shape = input.get_shape().as_list()
    input_length = input_shape[1]
    if not input_length:
        input_length = K.shape(inputs)[1]
    # Shape: (num_samples * timesteps, ...). And track the
    # transformation in self._input_map.
    input_uid = _object_list_uid(inputs)
    inputs = K.reshape(inputs, (-1,) + input_shape[2:])
    self._input_map[input_uid] = inputs
    # (num_samples * timesteps, ...)
    y = layer.call(inputs, **kwargs)
    if hasattr(y, '_uses_learning_phase'):
        uses_learning_phase = y._uses_learning_phase
    # Shape: (num_samples, timesteps, ...)
    output_shape = self.compute_output_shape(input_shape)
    y = K.reshape(y, (-1, input_length) + output_shape[2:])
    return y


class EncoderDecoder:
    """Encoder Decoder Network
    """
    def __init__(self,
                 num_freq_bins,
                 num_time_steps,
                 conv1_num_filters=64,
                 conv2_num_filters=96,
                 use_dropout=True,
                 spatial_dropout_rate=0.3,
                 random_seed=42,
                 max_to_keep=5):
        self.num_freq_bins = num_freq_bins
        self.num_time_steps = num_time_steps
        self.conv1_num_filters = conv1_num_filters
        self.conv2_num_filters = conv2_num_filters
        self.use_dropout = use_dropout,
        self.spatial_dropout_rate = spatial_dropout_rate,
        self.random_seed=random_seed

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder("float",
                                        shape=[None,
                                               self.num_freq_bins,
                                               None],
                                        name="input")
            self.output = tf.placeholder("int32",shape=[None])
            self.build_graph()
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_graph(self):
        input_reshape = tf.reshape(self.input,
                                   [-1,
                                    self.num_freq_bins
                                    - 1,
                                    1])
        with tf.name_scope('encoder'):
            encoder_layer1_filter_shape = [self.num_freq_bins,self.num_time_steps]
            encoder_layer1 = encoder_layer(input=input_reshape,
                                           num_filters=self.conv1_num_filters,
                                           filter_shape=encoder_layer1_filter_shape,
                                           padding='same',
                                           random_seed=self.random_seed,
                                           use_spatial_dropout=self.use_dropout,
                                           dropout_rate=self.spatial_dropout_rate,
                                           pool_size=self.pool_size,
                                           name='encode1')
            encoder_layer2_filter_shape = [self.num_freq_bins,self.num_time_steps]
            encoder_layer2 = encoder_layer(input=encoder_layer1,
                                           num_filters=self.conv2_num_filters,
                                           filter_shape=encoder_layer2_filter_shape,
                                           padding='same',
                                           random_seed=self.random_seed,
                                           use_spatial_dropout=self.use_dropout,
                                           dropout_rate=self.spatial_dropout_rate,
                                           pool_size=self.pool_size,
                                           name='encode2')

        with tf.name_scope('decoder'):
            decoder_layer1_filter_shape = [self.num_freq_bins,
                                           self.num_time_steps]
            decoder_layer1 = decoder_layer(encoder_layer2,
                                           num_filters=self.conv2_num_filters,
                                           filter_shape,
                                           height_factor=2,
                                           width_factor=2,
                                           padding='same',
                                           random_seed=42,
                                           use_spatial_dropout=self.use_dropout,
                                           dropout_rate=spatial_dropout_rate,
                                           name=None)
            decoder_layer2 = decoder_layer(decoder_layer1,
                                           num_filters=self.conv1_num_filters,
                                           filter_shape,
                                           height_factor=2,
                                           width_factor=2,
                                           padding='same',
                                           random_seed=42,
                                           use_spatial_dropout=self.use_dropout,
                                           dropout_rate=spatial_dropout_rate,
                                           name=None)
            out = time_distributed(decoder_layer_2)

