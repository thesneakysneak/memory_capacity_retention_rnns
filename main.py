from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

import generate_dataset as gd
import recurrent_models as mds


import random
from datetime import datetime

from datetime import datetime

import itertools

import database_functions

# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/


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

# Place holder
def data():
    return data.X_train,data.y_train,data.X_test,data.y_test

def investigate_number_of_patterns():
    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    network_type = ["lstm", "gru", "elman_rnn", "jordan_rnn"]
    # Variable we are investigating
    num_patterns = [x for x in range(2, 15)]

    # constants
    sequence_length = 1
    sparsity_length = 0
    for i in num_patterns:
        num_input_nodes = i
        print("     num_input_nodes ", i, "output_nodes", 2**num_input_nodes, "num_patterns", 2**num_input_nodes)

        # masters_user, password, experiment1_num_patterns
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=2**num_input_nodes,
                                  num_patterns=2**num_input_nodes,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)

        data.X_train = train_input
        data.y_train = train_out
        data.X_test = train_input
        data.y_test = train_out
        import recurrent_models
        g = recurrent_models.get_lstm
        g.num_input = num_input_nodes
        g.batch_size = 10
        g.timesteps = 1
        g.activation_function = "tanh"
        g.network_type = "lstm"
        g.num_output = 2**num_input_nodes

        best_run, best_model = optim.minimize(model=g,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials(), verbose=0)
        print(best_model)


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
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    pattern_input_set, random_patterns, input_set = gd.generate_set(num_input_nodes, sequence_length, num_patterns)
    pattern_output_set, random_output, output_set = gd.generate_set(num_output_nodes, 1, num_patterns)

    train_list, train_out = gd.create_equal_spaced_patterns(input_set, output_set, random_patterns, random_output, sparsity_length)

    # Test the effect of increasing number of patterns


    # Test the effect of increasing sparsity


    # Test the effect of increasing number of time steps

    return

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
    investigate_number_of_patterns()

if __name__ == "__main__":
    main()