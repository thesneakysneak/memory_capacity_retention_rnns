import random

import numpy
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, LambdaCallback
from keras.layers import SimpleRNN, Dense, Concatenate, SimpleRNNCell, constraints, regularizers, initializers, \
    activations, Bidirectional
import keras
from keras.engine.base_layer import Layer
from keras import backend as K
from keras.layers.recurrent import _generate_dropout_mask
from keras.legacy import interfaces
from keras.engine.base_layer import InputSpec
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow as tf

import numpy as np

import copy
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.utils.generic_utils import func_dump, CustomObjectScope
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.legacy import interfaces



class CustomDense(Dense):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.current_output = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)

        self.current_output = output
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

        self.go_backwards = None

        # https://github.com/keras-team/keras/blob/e24625095a33a5c9a2d016018203938e9bb2ccbf/keras/backend/tensorflow_backend.py#L2680
        if kwargs:
            if "go_backwards" in kwargs.keys():
                self.go_backwards = kwargs["go_backwards"]

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.output_layer_kernel = self.add_weight(shape=(self.next_layer.shape[-1].value, self.units),
                                                       name='jordan_kernel',
                                                       initializer=self.kernel_initializer,
                                                       regularizer=self.kernel_regularizer,
                                                       constraint=self.kernel_constraint)

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

    def set_next_layer(self, layer):
        K.set_value(self.next_layer, layer)

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
        # if self.next_layer != None:
        print("Jordan activate", prev_output.shape, self.output_layer_kernel.shape, self.next_layer, keras.backend.batch_dot(self.next_layer,
                                                                                         self.output_layer_kernel,
                                                                                         axes=None) )
        output = h + K.dot(prev_output, self.recurrent_kernel) + keras.backend.batch_dot(self.next_layer,
                                                                                         self.output_layer_kernel,
                                                                                         axes=None)  # K.dot()
        # else:
        #     print("Jordan not activate")
        #     output = h + K.dot(prev_output, self.recurrent_kernel)

        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]

    def set_next_layer(self, next_layer):
        """Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        """
        K.tf.assign(self.next_layer, next_layer)

    def get_next_layer_output(self):
        """Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        """
        params = self.next_layer
        return K.batch_get_value(params)

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
                  'recurrent_dropout': self.recurrent_dropout,
                  'go_backwards' : self.go_backwards
                  }
        base_config = super(SimpleRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class JordanCallback(Callback):
    def __init__(self, layers, cells_list, output_layer, model):
        self.layers = layers
        self.cells_list = cells_list
        self.output_layer = output_layer
        self.model = model
    # customize your behavior
    def on_batch_end(self, batch, logs=None):
        print("Assign", len(self.layers), len(self.cells_list))
        # print(K.eval(self.output_layer).reshape(-1,1,1).shape)
        # print(self.output_layer.dtype, K.eval(self.model.outputs[0]))

        # for l in range(len(self.layers) - 1):
        #     for i in self.cells_list[l]:
        #         K.tf.assign(i.next_layer, self.layers[l + 1])
        #         print("HAY", batch, K.get_value(i.next_layer), K.get_value(self.output_layer))
        #
        for i in self.cells_list[-1]:
            if self.model.outputs:
                # K.set_value(i.next_layer, np.array([[0.5]]))
                i.set_next_layer(self.output_layer)
        #         i.build((None, 1))
        #     else:
        #         print("NANI")
            # K.set_value(i.next_layer, np.array(K.batch_get_value(self.output_layer)).reshape(-1, 1))

            # print("HAY", batch, K.get_value(self.output_layer.output))

        # for i in self.cells_list[-1]:
        #     # print("Rebuilding")
        #     i.set_next_layer(self.output_layer)
        #     # i.build((None, 1))
        # a = K.get_value(self.output_layer)
        # print("AAAAAAAAAAAAAAAAAAAAAAA ", a)
        # if self.output_layer is not None and CustomDense(self.output_layer).current_output is not None:
        #     self.model.layers[1].cell.cells[0].set_next_layer(CustomDense(self.output_layer).current_output)



def build_jordan_layer(previous_layer, num_nodes_next_layer, num_nodes_in_layer, activation="tanh"):
    # n = tf.placeholder(tf.float32, shape=(None, num_nodes_next_layer), name="next_jordan_val")
    n = K.variable([[-1.0] * num_nodes_next_layer], name="next_jordan_val")

    output_layer = keras.Input(tensor=n, name="next_jordan_val_1")

    # with CustomObjectScope({'SimpleJordanRNNCell': SimpleJordanRNNCell}):
    cells = [Bidirectional(SimpleJordanRNNCell(previous_layer.get_shape()[-1].value, next_layer=output_layer, activation=activation) , return_sequences=True) for _ in
             range(num_nodes_in_layer)]

    layer = cells
    hidden_layer = layer(previous_layer)

    return hidden_layer, cells

def build_jordan_model(architecture=[],activation="tanh", bidirectional=False):
    input_layer = keras.Input((None, architecture[0]))
    layers = []
    cells_list = []
    next_middle_layer = None
    for i in range(1, len(architecture) - 1):
        if not next_middle_layer:
            # input layer
            layer, cells = build_jordan_layer(previous_layer=input_layer,
                                                   num_nodes_in_layer=architecture[i],
                                                   num_nodes_next_layer=architecture[i+1], activation=activation)
            layers.append(layer)

            cells_list.append(cells)
        else:
            layer, cells = build_jordan_layer(previous_layer=next_middle_layer,
                                                   num_nodes_in_layer=architecture[i],
                                                   num_nodes_next_layer=architecture[i+1], activation=activation)

            layers.append(layer)

            cells_list.append(cells)


    output_layer = CustomDense(architecture[-1])(layers[-1])
    for l in range(len(layers) - 1):
        for i in cells_list[l]:
            K.tf.assign(i.next_layer, layers[l + 1])


    # K.tf.assign(cells_list[-1][0].next_layer, output_layer)

    model = Model([input_layer], output_layer)
    model.Callback_var = JordanCallback(layers=layers, cells_list=cells_list, output_layer=output_layer, model=model)
    return model


def test():
    num_inputs = 1
    num_output_layer_outputs = 1
    x = [random.random() for i in range(2)]
    y = [random.random() for i in range(2)]
    x = y

    x = numpy.array(x).reshape(-1, 1, 1).astype(np.float32)
    y = numpy.array(y).reshape(-1, 1).astype(np.float32)

    num_cells_in_hidden_layer = 10

    input_layer = keras.Input((None, num_inputs))

    n = K.variable([[0.5]])

    output_layer = keras.Input(tensor=n)

    cells = [SimpleJordanRNNCell(num_inputs, next_layer=output_layer, activation="elu") for _ in
             range(num_cells_in_hidden_layer)]

    layer = keras.layers.RNN(cells)
    hidden_layer = layer(input_layer)
    output_layer = Dense(1)(hidden_layer)

    for i in cells:
        i.next_layer = output_layer

    model = Model([input_layer], output_layer)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                  patience=20, min_lr=0.000000001, verbose=1, cooldown=1)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    callbacks = [
        model.Callback_var,
        ModelCheckpoint(
            filepath="weights/weights-improvement-{epoch:02d}.hdf5",
            monitor="val_loss", verbose=1, save_best_only=False),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
    ]

    model.fit(x, y, epochs=1000, verbose=1, callbacks=callbacks)

    y_predict = model.predict(x)
    for i in range(len(y)):
        print(y_predict[i], y[i])


def test2():
    # sess = tf.Session()
    #
    # K.set_session(sess)

    num_inputs = 1
    num_output_layer_outputs = 1
    x = [random.random() for i in range(100)]
    y = [random.random() for i in range(100)]
    # x = y

    x = numpy.array(x).reshape(-1, 1, 1).astype(np.float32)
    y = numpy.array(y).reshape(-1, 1).astype(np.float32)

    num_cells_in_hidden_layer = 10

    #
    # model = build_jordan_model([num_inputs, num_cells_in_hidden_layer, num_output_layer_outputs], activation="tanh")
    #
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #
    # callbacks = [
    #     LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[1].cell.cells[0].get_next_layer_output())),
    #     model.Callback_var,
    #     ModelCheckpoint(
    #         filepath="weights/weights-improvement-{epoch:02d}.hdf5",
    #         monitor="val_loss", verbose=1, save_best_only=False),
    #     ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00000001)
    # ]
    #
    # model.fit(x, y, epochs=100, verbose=1, callbacks=callbacks, batch_size=10)
    #
    # y_predict = model.predict(x)
    # for i in range(len(y)):
    #     print(x[i], y_predict[i], y[i])
    #


    # Bidir Jordan
    model = build_jordan_model([num_inputs, num_cells_in_hidden_layer, num_output_layer_outputs], activation="tanh", bidirectional=True)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    callbacks = [
        LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[1].cell.cells[0].get_next_layer_output())),
        model.Callback_var,
        ModelCheckpoint(
            filepath="weights/weights-improvement-{epoch:02d}.hdf5",
            monitor="val_loss", verbose=1, save_best_only=False),
        ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]

    model.fit(x, y, epochs=100, verbose=1, callbacks=callbacks, batch_size=10)

    y_predict = model.predict(x)
    for i in range(len(y)):
        print(x[i], y_predict[i], y[i])




test2()




