#
# # LSTM for international airline passengers problem with time step regression framing
# import numpy
# import matplotlib.pyplot as plt
# from keras import Model
# from keras.utils import plot_model
# from pandas import read_csv
# import math
# from keras.models import Sequential
# from keras.layers import Dense, LSTMCell, add, TimeDistributed, Add, Concatenate
# from keras.layers import LSTM, Recurrent
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from keras.layers import Input
# import numpy as np
# sample_input = np.array([[[0],[1]],[[0],[1]]])
# sample_output = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
#
# #
# #
# #
# # Define an input sequence and process it.
# inputs = 5
# num_layer2_outputs = 2
# layer3_outputs = 2
#
# layer2_outputs = Input(shape=(None, num_layer2_outputs ), name='layer2_outputs')
# dense_output = Input(shape=(None, layer3_outputs), name='layer3_outputs')
#
# # Layer 1
#
# layer1_inputs = Input(shape=(None, inputs), name='layer1_input')
# layer1_prev_out = Concatenate(axis=-1)([layer1_inputs, layer2_outputs])
#
# layer1 = LSTM(7, return_state=True, return_sequences=True)
# layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)
#
#
# layer2_prev_out = layer1_outputs
#
# # layer2_prev_out = Concatenate()([layer1_outputs, dense_output])
#
# # Layer 2
# layer2 = LSTM(num_layer2_outputs , return_state=True, return_sequences=True)
# layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer1)
#
# # Layer 2
# layer3 = Dense(layer3_outputs, activation="softmax")
# dense_output = layer3(layer2_outputs)
#
#
# #
# # Build model
# #
#
# model = Sequential()
# model.add(layer1)
# model.add(layer2)
# # model.add(layer3)
#
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# input_set = np.array([[[1]*inputs + [0]*100], [[2]*inputs+ [0]*100], [[3]*inputs+ [0]*100]])
# output_set =np.array( [[0]*layer3_outputs, [1]*layer3_outputs, [1]*layer3_outputs,])
#
# model.fit(input_set, output_set, epochs=2, batch_size=1,  verbose=2)
# plot_model(model, show_shapes=True)




# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from keras import Model
from keras.utils import plot_model
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTMCell, add, TimeDistributed, Add, Concatenate, K
from keras.layers import LSTM, Recurrent
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input
import numpy as np
sample_input = np.array([[[1, 1, 1, 1, 1],[0, 0]],[[1, 1, 1, 1, 1],[0, 0]]])
sample_output = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])

#
#
#
# Define an input sequence and process it.
def single_layer_jordan_rnn(num_inputs, num_output_layer_outputs, num_nodes_layer1):

    dense_output = Input(shape=(None, num_output_layer_outputs ))

    # Layer 1
    layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
    layer1_prev_out = Concatenate()([layer1_inputs, dense_output])

    layer1 = LSTM(num_nodes_layer1, return_state=True, return_sequences=True, name='layer1')
    layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

    # Layer 2
    output_nodes = Dense(num_output_layer_outputs, activation="softmax", name='output_layer')(layer1_outputs)

    model = Model([layer1_inputs, dense_output], output_nodes)
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
    output_nodes = Dense(num_layer2_outputs, activation="softmax", name='output_layer')(layer2_outputs)

    model = Model([layer1_inputs, dense_output, dense_output_2], output_nodes)
    return model


#
# Build model
#


plot_model(model, to_file='model.png')


model.compile(loss='categorical_crossentropy', optimizer='adam')

input_set = np.array([np.array([np.array([1]*num_inputs)])])
output_set =np.array( [[[1]*num_layer2_outputs]])

for epoc_ in range(10):
    input_set = np.array([1] * num_inputs)
    try:
        np_dense_output = K.eval(dense_output)
    except:
        np_dense_output = []
        for i in range(input_set.shape[0]):
            np_dense_output = np.array([np.array([np.array([1]*num_layer2_outputs)])])
    print(np_dense_output)
    input_set_ = np.concatenate((input_set, np_dense_output), axis=-1)
    model.fit([input_set, np_dense_output], output_set, epochs=1, verbose=2)
plot_model(model, show_shapes=True)

