from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D, Conv1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    # Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization() (simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # TimeDistributedDense layer(wraper) :
    # This wrapper allows us to apply a layer to every temporal slice of an input:
    time_dense = TimeDistributed (Dense (output_dim) ) (bn_rnn)
    # Add softmax activation layer
    # Softmax converts a real vector to a vector of categorical probabilities.
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name="bn_rnn")(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_no_0 = GRU(units, activation='relu',return_sequences=True, implementation=2, name='frist_RNN')(input_data)
    rnn_prev = rnn_no_0
    # here: a for loop generating recur_layers ..:
    for layer in range(1 , recur_layers ):
        layer_name = "rnn_no_" +  str(layer)
        rnn = GRU(units, activation="relu",return_sequences=True, 
                 implementation=2, name=layer_name)(rnn_prev)
        # Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
        batch_name = "batch_no_" + str(layer)
        rnn_out = BatchNormalization(name=batch_name)(rnn)
        rnn_prev = rnn_out
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_out)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    # single bidirectional RNN layer, before a (TimeDistributed) dense layer ::
    # tf.keras.layers.Bidirectional(layer, merge_mode="concat", weights=None, backward_layer=None, **kwargs
    bidir_rnn = Bidirectional(GRU(units, activation="relu",return_sequences=True,implementation=2, name="bidir_rnn"))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed ( Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # 1D convolution layer
    # This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs
    conv_layer = Conv1D(filters, kernel_size, strides=conv_stride,padding=conv_border_mode, activation='relu',name='conv1d')(input_data) 
    # maxpool : Downsamples the input representation by taking the maximum value over the window defined by pool_size.
    maxpool_layer = MaxPooling1D(pool_size=9, strides=3, padding='valid')(conv_layer)
    # Batch Normalization
    conv1_normalized = BatchNormalization(name="maxpool_layer")(conv_layer)
    # Bidirectionnal RNN
    bidir_rnn_1 = Bidirectional(GRU(units, activation="relu",return_sequences=True,implementation=2, name="bidir_rnn"))(conv1_normalized)
    bidir_rnn_2 = Bidirectional(GRU(units, activation="relu",return_sequences=True,implementation=2, name="bidir_rnn"))(bidir_rnn_1)
    bidir_rnn_3 = Bidirectional(GRU(units, activation="relu",return_sequences=True,implementation=2, name="bidir_rnn"))(bidir_rnn_2) 
    # Batch Normalization :
    bidir_rnn_normalized = BatchNormalization(name="bidir_rnn_normalized")(bidir_rnn_3)
    # Time distributed :
    time_dense1 = TimeDistributed(Dense(output_dim))(bidir_rnn_normalized)
    # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
    dropout = Dropout(0.4) (time_dense1)
    # Time distributed
    time_dense2 = TimeDistributed(Dense(output_dim))(dropout)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # DONE: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    
    print(model.summary())
    return model