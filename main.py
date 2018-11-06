from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

import generate_dataset as gd
import recurrent_models
import recurrent_models as mds

import random
from datetime import datetime

from datetime import datetime

import itertools

import database_functions

# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

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


def search_architecture(num_input,
                        num_out,
                        x_train,
                        y_train,
                        batch_size=10,
                        timesteps=3,
                        network_type="lstm",
                        activation_function='tanh'):
    model = None
    for depth in range(num_input * 3):
        for l5 in range(depth):
            for l4 in range(depth):
                for l3 in range(depth):
                    for l2 in range(depth):
                        upper_bound = depth
                        lower_bound = 1
                        if depth > int(np.ceil(num_input/2)):
                            upper_bound = int(np.ceil(num_input/2))

                        for l1 in range(lower_bound, upper_bound):
                            # Stop if accuracy == 1

                            architecture = [num_input, l1, l2, l3, l4, l5, num_out]
                            print("architecture", architecture)
                            architecture = list(filter(lambda a: a != 0, architecture))  # remove 0s

                            model = recurrent_models.get_model(architecture=architecture,
                                                               batch_size=batch_size,
                                                               timesteps=timesteps,
                                                               network_type=network_type,
                                                               activation_function=activation_function)
                            model, result = recurrent_models.train_model(x_train, y_train, model, "adam",
                                                                         batch_size)
                            validation_acc = np.amax(result.history['acc'])
                            if validation_acc >= 1.0:
                                print('Best validation acc of epoch:', validation_acc, "architecture", architecture)
                                return model


    return model


# def get_lstm(x_train, y_train, x_test, y_test):

#     batch_size = 1
#     timesteps = 1
#     activation_function = "tanh"
#     network_type = "lstm"
#     num_input = get_lstm.num_input
#
#     architecture = {{ choice([1, 2, 3, 4]) }}
#     architecture = [architecture ]
#     model = Sequential()
#
#     return_sequences = False
#     if len(architecture) > 1:
#         return_sequences = True
#
#     if network_type == "lstm":
#         model.add(LSTM(architecture[0], batch_input_shape=(batch_size, timesteps, num_input),
#                        stateful=True, return_sequences=return_sequences, activation=activation_function))
#     elif network_type == "gru":
#         model.add(GRU(architecture[0], batch_input_shape=(batch_size, timesteps, num_input),
#                       stateful=True, return_sequences=return_sequences, activation=activation_function))
#     elif network_type == "elman_rnn":
#         model.add(SimpleRNN(architecture[0], batch_input_shape=(batch_size, timesteps, num_input),
#                       stateful=True, return_sequences=return_sequences, activation=activation_function))
#     elif network_type == "jordan_rnn":
#         model.add(SimpleRNN(architecture[0], batch_input_shape=(batch_size, timesteps, num_input),
#                       stateful=True, return_sequences=return_sequences, activation=activation_function))
#
#
#     # Hidden layer how many ever
#     for h in range(1, len(architecture)):
#         return_sequences = False
#         if h < len(architecture) - 1:
#             return_sequences = True
#
#         # print(h, return_sequences)
#         if network_type == "lstm":
#             model.add(LSTM(architecture[h], return_sequences=return_sequences, activation=activation_function))
#         elif network_type == "gru":
#             model.add(GRU(architecture[h], return_sequences=return_sequences, activation=activation_function))
#         elif network_type == "elman_rnn":
#             model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function))
#         elif network_type == "jordan_rnn":
#             model.add(SimpleRNN(architecture[h], return_sequences=return_sequences, activation=activation_function))
#
#     model.add(Dense(5, activation="softmax"))
#
#     import recurrent_models
#     callbacks = [
#         recurrent_models.earlystop,
#         recurrent_models.reset_state
#     ]
#
#     model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
#
#     # We can set shuffle to true by definition of the training set
#     result = model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2, shuffle=True, callbacks=callbacks)
#     #
#     # score, acc = model.evaluate(x_test, y_test, verbose=0)
#     #
#     # print('Test accuracy:', acc)
#     # return {'loss': -acc, 'status': STATUS_OK, 'model': model}
#     validation_acc = np.amax(result.history['acc'])
#     print('Best validation acc of epoch:', validation_acc)
#     return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
#


def single_experiment(engine,
                      case_type,
                      num_input_nodes,
                      num_output_nodes,
                      num_patterns,
                      sequence_length,
                      sparsity_length,
                      architecture,
                      network_type,
                      training_alg,
                      activation_function,
                      batch_size):
    # generate set
    train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
        gd.get_experiment_set(case_type=1,
                              num_input_nodes=num_input_nodes,
                              num_output_nodes=num_output_nodes,
                              num_patterns=num_patterns,
                              sequence_length=sequence_length,
                              sparsity_length=sparsity_length)
    # train model until converge
    # if model fails to fit, need to search the architecture
    model = mds.get_model(architecture=architecture,
                          batch_size=1,
                          timesteps=sequence_length,
                          network_type=network_type,
                          activation_function=activation_function)

    model = mds.train_model(train_input, train_out, model, training_alg=training_alg, batch_size=batch_size)
    return model.history.history["acc"][0]


def investigate_number_of_patterns():
    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    network_type = ["lstm", "gru", "elman_rnn", "jordan_rnn"]
    # Variable we are investigating
    num_patterns = [x for x in range(5, 10)]

    # constants
    sequence_length = 1
    sparsity_length = 0

    import recurrent_models
    from recurrent_models import get_lstm
    from recurrent_models import data

    for i in num_patterns:
        num_input_nodes = i
        print("     num_input_nodes ", i, "output_nodes", (2**num_input_nodes)**sequence_length, "num_patterns", (2**num_input_nodes)**sequence_length)

        # masters_user, password, experiment1_num_patterns
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=(2**num_input_nodes)**sequence_length,
                                  num_patterns=(2**num_input_nodes)**sequence_length,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)

        best_model = search_architecture(num_input_nodes,
                                         2 ** num_input_nodes,
                                         train_input,
                                         train_out,
                                         batch_size=10,
                                         timesteps=1,
                                         network_type="lstm",
                                         activation_function='tanh')
        print(best_model.summary())
        keras.backend.clear_session()


def run_experiments():
    """
    This function serves as the orchestration of experiments. As in all experiments, one wishes to keep all parameters
    constant except the variable being investigated. All results will be written to a database.
    :return:
    """
    # First get a neural network that is able to fit the input
    case_type = 1
    num_input_nodes = 1
    num_output_nodes = 4
    num_patterns = 1
    timesteps = 1
    sparsity_length = 0



    # Test the effect of increasing number of patterns


    # Test the effect of increasing sparsity


    # Test the effect of increasing number of time steps

    return

def test_generate_dataset():
    timesteps = [x for x in range(1, 10)]

    print("========================================================================================")
    print("========================================================================================")
    # Keep input size and num patterns constant. Increase timesteps
    for i in timesteps:
        num_input_nodes = 1
        sequence_length = i

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = 2
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    print("========================================================================================")
    print("========================================================================================")
    # Keep timesteps constant. Increase input size
    input_size = [x for x in range(1, 10)]

    for i in input_size:
        num_input_nodes = i
        sequence_length = 2

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = 2
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    # Keep input size and timesteps constant, increase num patterns
    print("========================================================================================")
    print("========================================================================================")


    for i in [x for x in range(2, 10)]:
        num_input_nodes = 10
        sequence_length = 1

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = i
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    # Keep input size,  num patterns and timesteps constant, increase sparsity
    print("========================================================================================")
    print("========================================================================================")

    for i in [x for x in range(2, 10)]:
        num_input_nodes = 10
        sequence_length = 1

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = i
        num_patterns = 3
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)

    print("train_input", train_input.shape)
    print("train_input", len(train_input))
    print("train_out", train_out.shape)
    print("train_input", len(train_out))

def test_loop():
    case_type = 1
    num_input_nodes = 1
    num_output_nodes = 4
    num_patterns = 1
    timesteps = 1
    sparsity_length = 0

    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    network_types = ["lstm", "gru", "elman_rnn", "jordan_rnn"]

    # Variable we are investigating
    for num_input_nodes in range(1, 10):
        for timesteps in range(1, 10):
            num_available_patterns = (2**num_input_nodes)**timesteps
            for num_patterns in range(1, num_available_patterns ):
                for sparsity_length in range(0, 100):
                    for network_type in network_types:
                        for activation_function in activation_functions:
                            for run in range(30):
                                df = database_functions.get_dataset(num_input_nodes, )


def main():
    case_type = 1
    num_input_nodes = 3
    num_output_nodes = 2
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    sparsity_erratic = 0
    random_seed = datetime.now().timestamp()
    binary_input = 1

    num_hidden_layers = 1
    network_type = "lstm"
    training_alg = "adam"
    activation_function = "tanh"
    architecture = [num_input_nodes, 2, num_output_nodes]
    batch_size = 10
    # gd.example()
    # investigate_number_of_patterns()
    test_generate_dataset()

if __name__ == "__main__":
    main()
