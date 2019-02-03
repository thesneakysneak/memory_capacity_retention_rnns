import keras
from keras import Model
from keras.utils import plot_model

from keras.layers import Dense, LSTMCell, add, TimeDistributed, Add, Concatenate, K, SimpleRNN, Reshape
from keras.layers import LSTM, Recurrent

from keras.layers import ZeroPadding3D
from keras.layers import Input
from keras.layers import RNN

import numpy as np
sample_input = np.array([[[1, 1, 1, 1, 1],[0, 0]],[[1, 1, 1, 1, 1],[0, 0]]])
sample_output = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])


class JordanRNNCell(keras.layers.Layer):

    def __init__(self, units, model, **kwargs):
        self.units = units
        self.state_size = units
        self.prev_model = model
        self.states = None
        super(JordanRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        if self.prev_model is not None and self.prev_model.states is not None:
            prev_output = self.prev_model.states[0]
        else:
            prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        self.states = states
        return output, [output]

# Layer2 = Dense(10)
# Layer1 = JordanRNNCell(10, Layer2)
# cells = [Layer1]
# x = keras.Input((None, 5))
# layer = RNN(cells)
# y = layer(x)
#
#
#
# Define an input sequence and process it.
def single_jordan_rnn(num_inputs, num_output_layer_outputs, num_nodes_layer1, input_shape=(1,1,1), batch_size=1, prev_model=None):
    #                           smaples, timesteps, features
    #
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    #
    if prev_model:
        prev_output_layer = prev_model.layers[1].output
    else:
        prev_output_layer = Input(shape=(None, num_output_layer_outputs), name='layer1_input_2')
    layer1_prev_out = Concatenate()([layer1_inputs, prev_output_layer])
    # Layer 1
    #
    layer1 = SimpleRNN(num_nodes_layer1, name='layer1')
    layer1_outputs = layer1(layer1_prev_out)
    #
    # Layer 2
    output = Dense(num_output_layer_outputs)(layer1_outputs)
    #
    model = Model([layer1_inputs, prev_output_layer], output)
    #
    if prev_model:
        model.set_weights(prev_model.get_weights())
    return model

def single_layer_jordan_rnn(num_inputs, num_output_layer_outputs, num_nodes_layer1, input_shape=(1,1,1), batch_size=1, prev_model=None):
    #                           smaples, timesteps, features
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')


    if prev_model:
        prev_output_layer = prev_model.layers[1].output
    else:
        prev_output_layer = Input(shape=(None, num_output_layer_outputs), name='layer1_input_2')
    layer1_prev_out = Concatenate()([layer1_inputs, prev_output_layer])
    # Layer 1

    layer1 = SimpleRNN(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h = layer1(layer1_prev_out)

    # Layer 2
    output_nodes = SimpleRNN(num_output_layer_outputs, return_state=True, return_sequences=True, name='output_layer')
    output_nodes_outputs, output_nodes_state_h = output_nodes(layer1_outputs)

    model = Model([layer1_inputs, prev_output_layer], output_nodes_outputs)

    if prev_model:
        model.set_weights(prev_model.get_weights())
    return model

def two_layer_jordan_rnn(num_inputs, num_layer2_outputs, num_output_layer_outputs, num_nodes_layer1, num_nodes_layer2):

    dense_output = Input(shape=(None, num_layer2_outputs ))
    dense_output_2 = Input(shape=(None, num_output_layer_outputs))

    # Layer 1
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    layer1_prev_out = Concatenate()([layer1_inputs, dense_output])

    layer1 = LSTM(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

    # Layer 2
    layer2_prev_out = Concatenate()([layer1_outputs, dense_output_2])
    layer2 = LSTM(num_nodes_layer2, return_state=True, return_sequences=True, name='layer2')
    layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer2_prev_out)

    ##########
    # output
    output_nodes = Dense(num_output_layer_outputs, activation="softmax", name='output_layer')(layer2_outputs)

    model = Model([layer1_inputs, dense_output, dense_output_2], output_nodes)
    return model

def three_layer_jordan_rnn(num_inputs,
                           num_layer2_outputs,
                            num_layer3_outputs,
                           num_output_layer_outputs,
                           num_nodes_layer1,
                           num_nodes_layer2,
                           num_nodes_layer3):

    dense_output = Input(shape=(None, num_layer2_outputs ))
    dense_output_2 = Input(shape=(None, num_layer3_outputs))
    dense_output_3 = Input(shape=(None, num_output_layer_outputs))

    # Layer 1
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    layer1_prev_out = Concatenate()([layer1_inputs, dense_output])

    layer1 = LSTM(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

    # Layer 2
    layer2_prev_out = Concatenate()([layer1_outputs, dense_output_2])
    layer2 = LSTM(num_nodes_layer2, return_state=True, return_sequences=True, name='layer2')
    layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer2_prev_out)

    # Layer 3
    layer3_prev_out = Concatenate()([layer2_outputs, dense_output_3])
    layer3 = LSTM(num_nodes_layer3, return_state=True, return_sequences=True, name='layer3')
    layer3_outputs, layer3_state_h, layer3_state_c = layer3(layer3_prev_out)

    ##########
    # output
    output_nodes = Dense(num_output_layer_outputs, activation="softmax", name='output_layer')(layer3_outputs)

    model = Model([layer1_inputs, dense_output, dense_output_2, dense_output_3], output_nodes)
    return model

def four_layer_jordan_rnn(num_inputs,
                            num_layer2_outputs,
                            num_layer3_outputs,
                            num_layer4_outputs,
                            num_output_layer_outputs,
                            num_nodes_layer1,
                            num_nodes_layer2,
                            num_nodes_layer3,
                            num_nodes_layer4):

    dense_output = Input(shape=(None, num_layer2_outputs ))
    dense_output_2 = Input(shape=(None, num_layer3_outputs))
    dense_output_3 = Input(shape=(None, num_layer4_outputs))
    dense_output_4 = Input(shape=(None, num_output_layer_outputs))

    # Layer 1
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    layer1_prev_out = Concatenate()([layer1_inputs, dense_output])

    layer1 = LSTM(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

    # Layer 2
    layer2_prev_out = Concatenate()([layer1_outputs, dense_output_2])
    layer2 = LSTM(num_nodes_layer2, return_state=True, return_sequences=True, name='layer2')
    layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer2_prev_out)

    # Layer 3
    layer3_prev_out = Concatenate()([layer2_outputs, dense_output_3])
    layer3 = LSTM(num_nodes_layer3, return_state=True, return_sequences=True, name='layer3')
    layer3_outputs, layer3_state_h, layer3_state_c = layer3(layer3_prev_out)

    # Layer 4
    layer4_prev_out = Concatenate()([layer3_outputs, dense_output_4])
    layer4 = LSTM(num_nodes_layer4, return_state=True, return_sequences=True, name='layer4')
    layer4_outputs, layer4_state_h, layer4_state_c = layer4(layer4_prev_out)

    ##########
    # output
    output_nodes = Dense(num_output_layer_outputs, activation="softmax", name='output_layer')(layer4_outputs)

    model = Model([layer1_inputs, dense_output, dense_output_2, dense_output_3], output_nodes)
    return model


def five_layer_jordan_rnn(num_inputs,
                            num_layer2_outputs,
                            num_layer3_outputs,
                            num_layer4_outputs,
                            num_layer5_outputs,
                            num_output_layer_outputs,
                            num_nodes_layer1,
                            num_nodes_layer2,
                            num_nodes_layer3,
                            num_nodes_layer4,
                            num_nodes_layer5):

    dense_output = Input(shape=(None, num_layer2_outputs ))
    dense_output_2 = Input(shape=(None, num_layer3_outputs))
    dense_output_3 = Input(shape=(None, num_layer4_outputs))
    dense_output_4 = Input(shape=(None, num_layer5_outputs))
    dense_output_5 = Input(shape=(None, num_output_layer_outputs))

    # Layer 1
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    layer1_prev_out = Concatenate()([layer1_inputs, dense_output])

    layer1 = LSTM(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

    # Layer 2
    layer2_prev_out = Concatenate()([layer1_outputs, dense_output_2])
    layer2 = LSTM(num_nodes_layer2, return_state=True, return_sequences=True, name='layer2')
    layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer2_prev_out)

    # Layer 3
    layer3_prev_out = Concatenate()([layer2_outputs, dense_output_3])
    layer3 = LSTM(num_nodes_layer3, return_state=True, return_sequences=True, name='layer3')
    layer3_outputs, layer3_state_h, layer3_state_c = layer3(layer3_prev_out)

    # Layer 4
    layer4_prev_out = Concatenate()([layer3_outputs, dense_output_4])
    layer4 = LSTM(num_nodes_layer4, return_state=True, return_sequences=True, name='layer4')
    layer4_outputs, layer4_state_h, layer4_state_c = layer4(layer4_prev_out)

    # Layer 5
    layer5_prev_out = Concatenate()([layer4_outputs, dense_output_5])
    layer5 = LSTM(num_nodes_layer5, return_state=True, return_sequences=True, name='layer5')
    layer5_outputs, layer5_state_h, layer5_state_c = layer5(layer5_prev_out)

    ##########
    # output
    output_nodes = Dense(num_output_layer_outputs, activation="softmax", name='output_layer')(layer5_outputs)

    model = Model([layer1_inputs, dense_output, dense_output_2, dense_output_3], output_nodes)
    return model


#
# Build model
#


#
# num_inputs = 10
# num_layer2_outputs = 2
# model = single_layer_jordan_rnn(num_inputs, num_layer2_outputs, 5,
#                                 input_shape=(1,1,1), batch_size=1, prev_model=None)
#
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# input_set = np.array([np.array([np.array([1]*num_inputs)])])
# prev_output_set = np.array( [[[1]*num_layer2_outputs]])
# output_set =np.array( [[[1]*num_layer2_outputs]])
# model.fit([input_set, prev_output_set], output_set, epochs=1, verbose=2)
#
#
#
#
#
# num_inputs = 10
# num_layer2_outputs = 2
# model2 = single_jordan_rnn(num_inputs, num_layer2_outputs, 5,
#                                 input_shape=(1,1,1), batch_size=1, prev_model=model)
#
# model2.compile(loss='categorical_crossentropy', optimizer='adam')
#
# input_set = np.array([np.array([np.array([1]*num_inputs)])])
# prev_output_set = np.array( [[[1]*num_layer2_outputs]])
# output_set =np.array( [[[1]*num_layer2_outputs]])
# model2.fit([input_set, prev_output_set], output_set, epochs=1, verbose=2)