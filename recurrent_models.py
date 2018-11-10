from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

import keras
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN

import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

import numpy as np
global x_train, y_train, x_test, y_test
x_train  = []
y_train = []
x_test= []
y_test = []


class ResetState(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.model.reset_states()
        # #         print("reset model state", logs)
        #         acc = logs.get("acc")
        #         if acc == 1.0:

        return

# Place holder
def data():
    from recurrent_models import data
    import recurrent_models
    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_train
    y_test = data.y_train
    return x_train, y_train, x_test, y_test


def get_lstm(x_train, y_train, x_test, y_test):
    from recurrent_models import get_lstm
    from recurrent_models import ResetState
    from recurrent_models import EarlyStopping
    import recurrent_models
    batch_size = get_lstm.batch_size
    timesteps = get_lstm.timesteps
    activation_function = get_lstm.activation_function
    network_type = get_lstm.network_type

    # Freaking weird serialization
    a = {{choice([i for i in range(0, get_lstm.num_input * 3)] + [0 for i in range(0, get_lstm.num_input * 3)])}}
    b = {{choice([i for i in range(0, get_lstm.num_input * 3)]+ [0 for i in range(0, get_lstm.num_input * 3)])}}
    c = {{choice([i for i in range(0, get_lstm.num_input * 3)]+ [0 for i in range(0, get_lstm.num_input * 3)])}}
    d = {{choice([i for i in range(0, get_lstm.num_input * 3)]+ [0 for i in range(0, get_lstm.num_input * 3)])}}
    e = {{choice([i for i in range(0, get_lstm.num_input * 3)]+ [0 for i in range(0, get_lstm.num_input * 3)])}}

    architecture = [a, b, c, d, e]
    architecture = list(filter(lambda a: a != 0, architecture))  # remove 0s

    return_sequences = False
    if len(architecture) > 1:
        return_sequences = True

    model = Sequential()


    if network_type == "lstm":
        model.add(LSTM(architecture[0], batch_input_shape=(batch_size, timesteps, get_lstm.num_input),
                       stateful=True,  return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "gru":
        model.add(GRU(architecture[0], batch_input_shape=(batch_size, timesteps, get_lstm.num_input),
                      stateful=True, return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "elman_rnn":
        model.add(SimpleRNN(architecture[0], batch_input_shape=(batch_size, timesteps, get_lstm.num_input),
                      stateful=True, return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "jordan_rnn":
        model.add(SimpleRNN(architecture[0], batch_input_shape=(batch_size, timesteps, get_lstm.num_input),
                      stateful=True, return_sequences=return_sequences, activation=activation_function, name="layer1"))


    # Hidden layer how many ever
    for h in range(1, len(architecture)):
        return_sequences = False
        if h < len(architecture) - 1:
            return_sequences = True

        # print(h, return_sequences)
        if network_type == "lstm":
            model.add(LSTM(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))
        elif network_type == "gru":
            model.add(GRU(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))
        elif network_type == "elman_rnn":
            model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))
        elif network_type == "jordan_rnn":
            model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))

    model.add(Dense(get_lstm.num_output+1, activation="softmax", name="out_layer"))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    earlystop = recurrent_models.EarlyStopping(monitor='loss',  # loss
                              patience=30,
                              verbose=1,
                              min_delta=0.05,
                              mode='auto')
    reset_state = recurrent_models.ResetState()

    callbacks = [
        earlystop,
        reset_state
    ]

    # We can set shuffle to true by definition of the training set
    result = model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, verbose=2, shuffle=True, callbacks=callbacks)
    #
    # score, acc = model.evaluate(x_test, y_test, verbose=0)
    # #
    # print('Test accuracy:', acc)
    # return {'loss': -acc, 'status': STATUS_OK, 'model': model}
    validation_acc = np.amax(result.history['acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_model(architecture=[2, 1, 1, 1],
              batch_size=10, timesteps=3,
              network_type="lstm",
              activation_function='tanh'):
    model = Sequential()
    # Hidden layer 1
    return_sequences = False
    unroll = False
    if len(architecture) > 3:
        return_sequences = True
        unroll = False
    if network_type == "lstm":
        model.add(LSTM(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                       stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "gru":
        model.add(GRU(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "elman_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function, name="layer1"))
    elif network_type == "jordan_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function, name="layer1"))

    # Hidden layer how many ever
    for h in range(2, len(architecture) - 1):
        print(h)

        return_sequences = False
        if h < len(architecture) - 2:
            return_sequences = True
        if network_type == "lstm":
            model.add(LSTM(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))

        elif network_type == "gru":
            model.add(GRU(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))
        elif network_type == "elman_rnn":
            model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function, name="layer"+str(h+1)))

    model.add(Dense(architecture[-1], activation="softmax", name="output_layer"))
    return model


def train_model(input_set, output_set, model, training_alg, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer=training_alg, metrics=['accuracy'])

    callbacks = [
        earlystop,
        reset_state
    ]

    result = model.fit(input_set, output_set, epochs=1000, batch_size=batch_size, verbose=1, shuffle=False, callbacks=callbacks)
    return model, result


def test():
    network_types = ["lstm", "gru", "elman_rnn", "jordan_rnn"]
    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    architecture = []
    import random
    for i in range(100):
        architecture = []
        num_layers = random.randint(1, 5)
        timesteps = random.randint(1, 300)
        batch_size = random.randint(1, 30)
        nn_index = random.randint(0, len(network_types) - 1)
        activation_index = random.randint(0, len(activation_functions) - 1)
        for i in range(num_layers):
            nodes_in_layer = random.randint(3, 1000)
            architecture.append(nodes_in_layer)
        print("architecture", architecture,
              "batch_size", batch_size,
              "timesteps", timesteps,
              "network_type", network_types[nn_index],
              "activation_function", activation_functions[activation_index])
        model = get_model(architecture=architecture,
                          batch_size=batch_size,
                          timesteps=timesteps,
                          network_type=network_types[nn_index],
                          activation_function=activation_functions[activation_index])
        print(model.summary())


def determine_score(predicted, test, f_only=True):
    p_categories = [np.argmax(x) for x in predicted]
    t_categories = [np.argmax(x) for x in test]

    conf_mat = confusion_matrix(t_categories, p_categories)
    precision, recall, fbeta_score, beta = precision_recall_fscore_support(t_categories, p_categories, average="micro")

    print(conf_mat)
    if f_only:
        return fbeta_score
    return precision, recall, fbeta_score, conf_mat


earlystop = EarlyStopping(monitor='accuracy',  # loss
                          patience=30,
                          verbose=1,
                          min_delta=0.05,
                          mode='auto')
reset_state = ResetState()

if __name__ == "main":
    test()
