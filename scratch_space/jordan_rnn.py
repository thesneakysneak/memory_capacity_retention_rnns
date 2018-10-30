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
sample_input = np.array([[[0],[1]],[[0],[1]]])
sample_output = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])

#
#
#
# Define an input sequence and process it.
num_inputs = 5
num_layer2_outputs = 2

dense_output = Input(shape=(None, num_layer2_outputs ), name='layer2_outputs')

# Layer 1
layer1_inputs = Input(shape=(None, num_inputs), name='layer1_input')
layer1_prev_out = Concatenate(axis=-1)([layer1_inputs, dense_output])

layer1 = LSTM(7, return_state=True, return_sequences=True, name='layer1')
layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_prev_out)

# Layer 2
output_nodes = Dense(num_layer2_outputs, activation="softmax", name='output_layer')
dense_output = output_nodes(layer1_outputs)


#
# Build model
#

model = Model([layer1_inputs], dense_output)
model.add(layer1)
model.add(output_nodes)

model.compile(loss='categorical_crossentropy', optimizer='adam')

input_set = np.array([[[1]*num_inputs], [[2]*num_inputs], [[3]*num_inputs]])
output_set =np.array( [[0]*num_layer2_outputs, [1]*num_layer2_outputs, [1]*num_layer2_outputs])

for epoc_ in range(10):
    input_set = np.array([[[1] * num_inputs]])
    try:
        np_dense_output = K.eval(dense_output)
    except:
        np_dense_output = []
        for i in range(input_set.shape[0]):
            np_dense_output.append([[0]*num_layer2_outputs])
    print(np_dense_output)
    input_set_ = np.concatenate((input_set, np_dense_output), axis=-1)
    model.fit(input_set_, output_set, epochs=1, verbose=2)
plot_model(model, show_shapes=True)


##############
#   Encoder
############
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)