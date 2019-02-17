import random

import numpy
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import SimpleRNN, Dense, Concatenate, SimpleRNNCell, constraints, regularizers, initializers
import keras
from keras.engine.base_layer import Layer
from keras import backend as K
from keras.layers.recurrent import _generate_dropout_mask
from keras.legacy import interfaces
from keras.engine.base_layer import InputSpec
from tensorboard.backend.event_processing.event_file_inspector import FLAGS
from tensorflow.python.keras.callbacks import TensorBoard


class SimpleJordanRNNCell(SimpleRNNCell):
    """Cell class for SimpleRNN.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 next_layer=None,
                 **kwargs):
        super(SimpleRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = keras.initializers.get(recurrent_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = keras.constraints.get(recurrent_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        # Jordan specific implementation
        self.next_layer = next_layer

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.next_layer != None:
            self.output_layer_kernel = self.add_weight(shape=(self.next_layer.shape[-1].value, self.units),
                                          name='jordan_kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            self.output_layer_kernel = None

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(prev_output),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask

        # Jordan specific output calculation
        if self.next_layer != None:
            # print("Jordan activate", prev_output.shape, self.output_layer_kernel.shape, self.next_layer.shape)
            output = h + K.dot(prev_output, self.recurrent_kernel) + keras.backend.batch_dot(self.next_layer, self.output_layer_kernel, axes=None) #K.dot()
        else:
            print("Jordan not activate")
            output = h + K.dot(prev_output, self.recurrent_kernel)

        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]

    def get_config(self):
        config = {'units': self.units,
                  'activation': keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      keras.initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      keras.initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      keras.regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      keras.regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      keras.constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






num_inputs = 1
num_output_layer_outputs = 1
x = [random.random() for i in range(10)]
y = [random.random() for i in range(10)]
x =y

x = numpy.array(x).reshape(-1, 1, 1).astype(np.float32)
y = numpy.array(y).reshape(-1, 1).astype(np.float32)

num_cells_in_hidden_layer = 10

input_layer = keras.Input((None, num_inputs))

n = K.variable([[-0.0]])

output_layer = keras.Input(tensor=n)

cells = [SimpleJordanRNNCell(num_inputs, next_layer=output_layer, activation="elu") for _ in range(num_cells_in_hidden_layer )]

layer = keras.layers.RNN(cells)
hidden_layer = layer(input_layer)
output_layer = Dense(1)(hidden_layer)

for i in cells:
    i.next_layer = output_layer



model = Model([input_layer], output_layer)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=20, min_lr=0.000000001, verbose=1, cooldown=1)

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])

callbacks = [
		ModelCheckpoint(filepath="/home/danielp/Documents/Masters/Code/memory_capacity_retention_rnns/scratch_space/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5",
                        monitor="loss", verbose=1, save_best_only=False),
		ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
	]

model.fit(x, y, epochs=1000, verbose=1, callbacks=[reduce_lr])


y_predict = model.predict(x)
for i in range(len(y)):
    print(y_predict[i], y[i])


