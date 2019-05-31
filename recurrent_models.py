from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import keras
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN

import generic_functions as gf

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

import tensorflow as tf
import numpy as np

global x_train, y_train, x_test, y_test
x_train = []
y_train = []
x_test = []
y_test = []


# def determine_score(predicted, test, f_only=True):
#     predicted = predicted.round()
#     test = test.round()
#     p_categories = [np.argmax(x) for x in predicted]
#     t_categories = [np.argmax(x) for x in test]
#     # for i in range(len(p_categories)):
#     #     if p_categories[i] != t_categories[i]:
#     #         print("Class not correct", p_categories[i], t_categories[i])
#     conf_mat = confusion_matrix(t_categories, p_categories)
#     precision, recall, fbeta_score, beta = precision_recall_fscore_support(t_categories, p_categories, average="micro")
#     # print(conf_mat)
#     if f_only:
#         return fbeta_score
#     return precision, recall, fbeta_score, conf_mat


class EarlyStopByF1(keras.callbacks.Callback):
    def __init__(self, value=0, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose
        self.prev_delta_score = 0.0
        self.delta_score = 0.0
        self.patience = 0

    def on_epoch_end(self, epoch, logs={}):

        predict = np.asarray(self.model.predict(self.validation_data[0], batch_size=10))
        target = self.validation_data[1]
        score = 0.0
        if len(predict[0]) > 1:
            score = gf.determine_ave_f_score(predict, target)
        else:
            score = gf.determine_f_score(predict, target)
        self.delta_score = score - self.prev_delta_score
        self.prev_delta_score = score

        # print("Epoch %05d: delta_score" % epoch, score, self.delta_score, self.patience)
        if np.abs(self.delta_score) < 0.05:
            self.patience += 1
        else:
            self.patience = 0

        if self.patience >= 700 or score > 0.98:
            if self.verbose > 0:
                print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True


class EarlyStopByF1OneHot(keras.callbacks.Callback):
    def __init__(self, value=0, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose
        self.prev_delta_score = 0.0
        self.delta_score = 0.0
        self.patience = 0

    def on_epoch_end(self, epoch, logs={}):

        predict = np.asarray(self.model.predict(self.validation_data[0], batch_size=10))
        target = self.validation_data[1]
        score = 0.0

        y_true = [np.argmax(x) for x in target]
        y_predict_unscaled = [np.argmax(x) for x in predict]
        score = gf.determine_f_score(y_predict_unscaled, y_true)
        self.delta_score = score - self.prev_delta_score
        self.prev_delta_score = score

        # print("Epoch %05d: delta_score" % epoch, score, self.delta_score, self.patience)
        if np.abs(self.delta_score) < 0.05:
            self.patience += 1
        else:
            self.patience = 0

        if self.patience >= 700 or score > 0.98:
            if self.verbose > 0:
                print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True


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
        # self.model.reset_states()
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


def get_model(architecture=[2, 1, 1, 1],
              batch_size=10,
              timesteps=3,
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
                       stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function,
                       name="layer1"))
    elif network_type == "gru":
        model.add(GRU(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function,
                      name="layer1"))
    elif network_type == "elman_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function,
                      name="layer1"))
    elif network_type == "jordan_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=unroll, return_sequences=return_sequences, activation=activation_function,
                      name="layer1"))

    # Hidden layer how many ever
    for h in range(2, len(architecture) - 1):
        print(h)

        return_sequences = False
        if h < len(architecture) - 2:
            return_sequences = True
        if network_type == "lstm":
            model.add(LSTM(architecture[h], return_sequences=return_sequences, activation=activation_function,
                           name="layer" + str(h + 1)))

        elif network_type == "gru":
            model.add(GRU(architecture[h], return_sequences=return_sequences, activation=activation_function,
                          name="layer" + str(h + 1)))
        elif network_type == "elman_rnn":
            model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function,
                                name="layer" + str(h + 1)))

    model.add(Dense(architecture[-1], activation="softmax", name="output_layer"))
    return model


def train_model(input_set, output_set, model, training_alg, batch_size, use_early_stop=False, verbose=0):
    model.compile(loss='mean_squared_error', optimizer=training_alg, metrics=['accuracy'])
    callbacks = []
    if use_early_stop:
        callbacks = [
            earlystop,
            earlystop2,
            reduce_lr,
            reset_state
        ]

    print("training")
    result = model.fit(input_set, output_set, epochs=1000, batch_size=batch_size, verbose=verbose,
                       shuffle=False, validation_data=(input_set, output_set), callbacks=callbacks)
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


earlystop2 = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=0, mode='auto')

earlystop = EarlyStopByF1OneHot(value=.99, verbose=1) #EarlyStopByF1(value=.99, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

reset_state = ResetState()

if __name__ == "main":
    test()
