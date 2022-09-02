# adapted from [1]_
# and https://github.com/colincsl/TemporalConvolutionalNetworks
#
# [1]René, Colin Lea Michael D. Flynn, and Vidal Austin Reiter Gregory D. Hager.
# "Temporal convolutional networks for action segmentation and detection." (2017).


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Add, Multiply, Dense, Reshape
from tensorflow.keras.layers import (
    Activation, 
    SpatialDropout1D,
    Lambda,
    Conv1D, 
    Conv2D, 
    ZeroPadding1D,
    Cropping1D,
    MaxPooling1D, 
    MaxPooling2D, 
    UpSampling1D,
    Bidirectional,
    LSTM
)

from keras import regularizers

import tensorflow as tf
from keras import backend as K

from tensorflow.keras.activations import relu
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
    return Multiply()([tanh_out, sigm_out])

def pool_output(input, pool_size, stride):
    return ((input - pool_size) // stride) + 1

def upsample_output(input, size):
    return input * size

def ed_tcn_output_size(input, n_nodes, pool_size=2):
    output = input
    for encoder_layer in range(len(n_nodes)):
        output = pool_output(output, pool_size, pool_size)
    for decoder_layer in range(len(n_nodes)):
        output = upsample_output(output, pool_size)
    return output


# models
def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', causal=False,
           optimizer="rmsprop", activation='norm_relu'):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[i], conv_len, padding='same',
                       kernel_regularizer=regularizers.l2(0.001))(model)
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
        model = Conv1D(n_nodes[-i - 1], conv_len, padding='same',
                       kernel_regularizer=regularizers.l2(0.001))(model)
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
    model = TimeDistributed(Dense(n_classes, activation="softmax",
                                  kernel_regularizer=regularizers.l2(0.001)))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
               n_filters_by_layer=(32, 64),
               loss='categorical_crossentropy', causal=False,
               optimizer="rmsprop", activation='norm_relu'):

    inputs = Input(shape=(max_len, n_feat))
    # add "channel" for conv2d layers
    model = inputs
    new_shape = (max_len, n_feat, 1)
    model = Reshape(new_shape)(model)

    for n_filters in n_filters_by_layer:
        model = Conv2D(n_filters, kernel_size=(5, 5),
                       kernel_regularizer=regularizers.l2(0.001),
                       padding="same", activation='relu')(model)
        model = MaxPooling2D(pool_size=(1, 8),
                         strides=(1, 8))(model)
    shape = model.get_shape().as_list()
    new_shape = [shape[1], shape[2] * shape[3]]
    # combine filters into one axis so we can apply 1d convolution
    model = Reshape(new_shape)(model)

    n_layers = len(n_nodes)

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[i], conv_len, padding='same',
                       kernel_regularizer=regularizers.l2(0.001))(model)
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
        model = Conv1D(n_nodes[-i - 1], conv_len, padding='same',
                       kernel_regularizer=regularizers.l2(0.001))(model)
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
    model = TimeDistributed(Dense(n_classes, activation="softmax",
                                  kernel_regularizer=regularizers.l2(0.001)))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks,
                max_len, activation="wavenet", tail_conv=1,
                use_skip_connections=True, causal=False,
                optimizer='adam', loss='categorical_crossentropy'):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2 ** i) // 2, 0))(x)
            conv = Conv1D(nb_filters, 2, dilation_rate=2 ** i,
                          padding='same',
                          name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
            conv = Cropping1D((0, (2 ** i) // 2))(conv)
        else:
            conv = Conv1D(nb_filters, 3, dilation_rate=2 ** i,
                          padding='same',
                          name='dilated_conv_%d_tanh_s%d' % (2 ** i, s),
                          kernel_regularizer=regularizers.l2(0.001))(x)

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
        x = Conv1D(nb_filters, 1, padding='same',
                   kernel_regularizer=regularizers.l2(0.001))(x)

        res_x = Add()([original_x, x])

        # return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))

    skip_connections = []

    x = input_layer
    if causal:
        x = ZeroPadding1D((1, 0))(x)
        x = Conv1D(nb_filters, 2, padding='same',
                   name='initial_conv')(x)
        x = Cropping1D((0, 1))(x)
    else:
        x = Conv1D(nb_filters, 3, padding='same', name='initial_conv',
                   kernel_regularizer=regularizers.l2(0.001))(x)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = Add()(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(nb_filters, tail_conv, padding='same',
               kernel_regularizer=regularizers.l2(0.001))(x)
    x = Activation('relu')(x)
    x = Conv1D(num_classes, tail_conv, padding='same',
               kernel_regularizer=regularizers.l2(0.001))(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    return model


def CNN_Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks,
                    max_len, n_filters_by_layer=(32, 64),activation="wavenet",
                    tail_conv=1, use_skip_connections=True, causal=False,
                    optimizer='adam', loss='categorical_crossentropy'):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2 ** i) // 2, 0))(x)
            conv = Conv1D(nb_filters, 2, dilation_rate=2 ** i,
                          padding='same',
                          name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
            conv = Cropping1D((0, (2 ** i) // 2))(conv)
        else:
            conv = Conv1D(nb_filters, 3, dilation_rate=2 ** i,
                          padding='same',
                          name='dilated_conv_%d_tanh_s%d' % (2 ** i, s),
                          kernel_regularizer=regularizers.l2(0.001))(x)

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
        x = Conv1D(nb_filters, 1, padding='same',
                   kernel_regularizer=regularizers.l2(0.001))(x)

        res_x = Add()([original_x, x])

        # return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))
    # add "channel" for conv2d layers
    x = input_layer
    new_shape = (max_len, num_feat, 1)
    x = Reshape(new_shape)(x)

    for n_filters in n_filters_by_layer:
        x = Conv2D(n_filters, kernel_size=(5, 5),
                   kernel_regularizer=regularizers.l2(0.001),
                   padding="same", activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 8),
                         strides=(1, 8))(x)
    shape = x.get_shape().as_list()
    new_shape = [shape[1], shape[2] * shape[3]]
    # combine filters into one axis so we can apply 1d convolution
    x = Reshape(new_shape)(x)

    skip_connections = []

    if causal:
        x = ZeroPadding1D((1, 0))(x)
        x = Conv1D(nb_filters, 2, padding='same',
                   name='initial_conv')(x)
        x = Cropping1D((0, 1))(x)
    else:
        x = Conv1D(nb_filters, 3, padding='same', name='initial_conv',
                   kernel_regularizer=regularizers.l2(0.001))(x)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = Add()(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(nb_filters, tail_conv, padding='same',
               kernel_regularizer=regularizers.l2(0.001))(x)
    x = Activation('relu')(x)
    x = Conv1D(num_classes, tail_conv, padding='same',
               kernel_regularizer=regularizers.l2(0.001))(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    return model


def CNN_biLSTM(n_classes, n_feat, max_len,
               n_filters_by_layer=(32, 64),
               loss='categorical_crossentropy',
               optimizer='Adam'):
    inputs = Input(shape=(max_len, n_feat))

    x = inputs
    new_shape = (max_len, n_feat, 1)
    x = Reshape(new_shape)(x)
    for n_filters in n_filters_by_layer:
        x = Conv2D(n_filters, kernel_size=(5, 5),
                   kernel_regularizer=regularizers.l2(0.001),
                   padding="same", activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 8),
                         strides=(1, 8))(x)
    conv_out_shape = x.get_shape().as_list()
    num_hidden = conv_out_shape[2] * conv_out_shape[3]
    new_shape = (-1, num_hidden)
    x = Reshape(new_shape)(x)
    x = Bidirectional(
        LSTM(num_hidden, return_sequences=True, dropout=0.25,
             recurrent_dropout=0.1))(x)
    x = TimeDistributed(Dense(n_classes, activation="softmax"))(x)
    model = Model(inputs=inputs, outputs = x)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
