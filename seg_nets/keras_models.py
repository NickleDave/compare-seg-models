# adapted from [1]_
# and https://github.com/colincsl/TemporalConvolutionalNetworks
#
# [1]René, Colin Lea Michael D. Flynn, and Vidal Austin Reiter Gregory D. Hager.
# "Temporal convolutional networks for action segmentation and detection." (2017).

from keras.models import Model
from keras.layers import Input, TimeDistributed
from keras.layers.core import *
from keras.layers.convolutional import *

import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial

clipped_relu = partial(relu, max_value=5)


# utility functions
def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return Merge(mode='mul')([tanh_out, sigm_out])


# models
def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', causal=False,
           optimizer="rmsprop", activation='norm_relu',
           return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization,
                           name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Convolution1D(n_nodes[-i - 1], conv_len, border_mode='same')(
            model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization,
                           name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal",
                  metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks,
                max_len,
                activation="wavenet", tail_conv=1, use_skip_connections=True,
                causal=False,
                optimizer='adam', return_param_str=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2 ** i) // 2, 0))(x)
            conv = AtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i,
                                       border_mode='same',
                                       name='dilated_conv_%d_tanh_s%d' % (
                                       2 ** i, s))(x)
            conv = Cropping1D((0, (2 ** i) // 2))(conv)
        else:
            conv = AtrousConvolution1D(nb_filters, 3, atrous_rate=2 ** i,
                                       border_mode='same',
                                       name='dilated_conv_%d_tanh_s%d' % (
                                       2 ** i, s))(x)

        conv = SpatialDropout1D(0.3)(conv)
        # x = WaveNet_activation(conv)

        if activation == 'norm_relu':
            x = Activation('relu')(conv)
            x = Lambda(channel_normalization)(x)
        elif activation == 'wavenet':
            x = WaveNet_activation(conv)
        else:
            x = Activation(activation)(conv)

            # res_x  = Convolution1D(nb_filters, 1, border_mode='same')(x)
        # skip_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
        x = Convolution1D(nb_filters, 1, border_mode='same')(x)

        res_x = Merge(mode='sum')([original_x, x])

        # return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))

    skip_connections = []

    x = input_layer
    if causal:
        x = ZeroPadding1D((1, 0))(x)
        x = Convolution1D(nb_filters, 2, border_mode='same',
                          name='initial_conv')(x)
        x = Cropping1D((0, 1))(x)
    else:
        x = Convolution1D(nb_filters, 3, border_mode='same',
                          name='initial_conv')(x)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = Merge(mode='sum')(skip_connections)
    x = Activation('relu')(x)
    x = Convolution1D(nb_filters, tail_conv, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution1D(num_classes, tail_conv, border_mode='same')(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(input_layer, x)
    model.compile(optimizer, loss='categorical_crossentropy',
                  sample_weight_mode='temporal')

    if return_param_str:
        param_str = "D-TCN_C{}_B{}_L{}".format(2, nb_stacks, dilation_depth)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model
