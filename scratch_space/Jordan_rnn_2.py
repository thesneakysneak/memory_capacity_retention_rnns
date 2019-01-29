import random

import numpy
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import SimpleRNN, Dense, Concatenate

num_inputs = 1
x = [random.random() for i in range(50)]
y = [random.random() for i in range(50)]


x = numpy.array(x).reshape(-1, 1).astype(np.float32)
y = numpy.array(y).reshape(-1, 1).astype(np.float32)

inp = Input(shape=x.shape, name="input_layer")
prev_out = Input(shape=y.shape, name="previous_output")

layer1_inputs = Input(shape=x.shape, name='layer1_input')
#

prev_model = None
if prev_model:
    prev_output_layer = prev_model.layers[1].output
else:
    prev_output_layer = Input(shape=y.shape, name='layer1_input_2')
layer1_prev_out = Concatenate()([layer1_inputs, prev_output_layer])
# Layer 1
#
layer1 = SimpleRNN(10, name='layer1')
layer1_outputs = layer1(layer1_prev_out)
#
# Layer 2
output = Dense(1)(layer1_outputs)
#
model = Model([layer1_inputs, prev_output_layer], output)

model.compile(loss='categorical_crossentropy', optimizer='adam')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.0000001)

initial_output_states = np.array([0] * 50).reshape(-1, 1).astype(np.float32)

# TODO Decide if you want to change the input shape to contain both the y_predicted and x
# For now I am going to revisit the jordan cell
model.fit([x, initial_output_states], y, validation_split=.2, callbacks=[reduce_lr])









