
# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTMCell, add, TimeDistributed
from keras.layers import LSTM, Recurrent
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input

input = Input((10,))
readout_input = Input((10,))
h_tm1 = Input((10,))
c_tm1 = Input((10,))

lstm_input = add([input, readout_input]) # Here we add to input.. you can do whatever you want with a Lambda layer

output, h_t, c_t = LSTMCell(10)([lstm_input, h_tm1, c_tm1])

rnn = Recurrent(input=input, initial_states=[h_tm1, c_tm1], output=output, final_states=[h_t, c_t], readout_input=readout_input)



#
#
#
# Define an input sequence and process it.
inputs = 5
layer2_outputs = 5
layer3_outputs = 2

# Layer 1
layer1_inputs = Input(shape=(None, inputs+layer2_outputs))
layer1 = LSTM(7, return_state=True)
layer1_outputs, layer1_state_h, layer1_state_c = layer1(layer1_inputs)

# Layer 2
layer2_inputs = Input(shape=(None, layer1_outputs+layer3_outputs))
layer2 = LSTM(layer2_outputs, return_state=True)
layer2_outputs, layer2_state_h, layer2_state_c = layer2(layer2_inputs)

# Layer 2
dense_output = TimeDistributed(layer3_outputs, activation="softmax")


#
# Build model
#

model = Sequential()
model.add(layer1)
model.add(layer2)
model.add(dense_output)
model.compile(loss='categorical_crossentropy', optimizer='adam')