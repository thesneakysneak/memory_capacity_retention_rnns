import logging
import os
import random
import sys

import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, SimpleRNN, GRU, Concatenate, LSTM

import experiment_constants as const
import recurrent_models
import generic_functions as gf
from scratch_space.jordan_rnn import JordanRNNCell

'''
    Define a random set of unique input to output mappings. No input output pair will correspond to other samples.
    Thus there will never exists a set of input values that will have the same output value.
    E.g. Correct
        0 -> 1
        1 -> 5
        2 -> 3
    E.g. Incorrect
        0 -> 1
        1 -> 1
        2 -> 1
'''


def generate_sets(num_patterns, scaled=True):
    correlated = True
    x = y = None
    while correlated:
        x = random.sample(range(1, num_patterns + 1), num_patterns)
        y = random.sample(range(1, num_patterns + 1), num_patterns)
        # print(x, y)
        if num_patterns == 1:
            correlated = False
        else:
            correlated = gf.are_sets_correlated(x, y)

    #
    if scaled:
        x = [1.0 / z for z in x]
        y = [1.0 / z for z in y]

    #
    training_set = list(zip(x, y))
    training_set = training_set * 100
    random.shuffle(training_set)
    #
    test_set = list(zip(x, y))
    test_set = test_set * 10
    random.shuffle(test_set)
    #
    x_train, y_train = zip(*training_set)
    x_test, y_test = zip(*test_set)
    #
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 1, 1).astype(np.float32)
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    #
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 1, 1).astype(np.float32)
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    #
    return x_train, y_train, x_test, y_test


def generate_sets_class(num_patterns):
    x = random.sample(range(1, num_patterns + 1), num_patterns)
    y = np.eye(num_patterns)
    #
    x = [1.0 / z for z in x]
    #
    training_set = list(zip(x, y))
    training_set = training_set * 100
    random.shuffle(training_set)
    #
    test_set = list(zip(x, y))
    test_set = test_set * 10
    random.shuffle(test_set)
    #
    x_train, y_train = zip(*training_set)
    x_test, y_test = zip(*test_set)
    #
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 1, 1).astype(np.float32)
    y_train = y_train.reshape(-1, num_patterns).astype(np.float32)
    #
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 1, 1).astype(np.float32)
    y_train = y_train.reshape(-1, num_patterns).astype(np.float32)
    #
    return x_train, y_train, x_test, y_test


def run_num_patterns(total_num_parameters=[1, 2], runner=1, thread=1):
    activation_functions = ["softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "relu",
                            "linear"]
    network_types = [const.LSTM, const.GRU, const.ELMAN_RNN,
                     const.BIDIRECTIONAL_RNN, const.BIDIRECTIONAL_LSTM,
                     const.BIDIRECTIONAL_GRU]  # "jordan_rnn" const.JORDAN_RNN

    run = runner
    logfile_location = "danny_masters"
    logfile = logfile_location + "/" + str(thread) + "_" + str(run) + "_num_patterns_one_hot.log"
    logfile = os.path.abspath(logfile)

    if not os.path.exists(logfile):
        f = open(logfile, "w")
        f.write(
            "nn_type;activation_func;parameters;nodes_in_layer;largest_retained;smallest_not_retained;model_params;num_epochs\n")
        f.close()

    logging.basicConfig(filename=logfile, level=logging.INFO)

    start = 2
    prev = 1
    steps = 0
    smallest_not_retained = 10000
    largest_retained = 0
    nodes_in_layer = 2
    activation_func = "sigmoid"
    nn_type = "lstm"

    num_divisible_by_all = 5040
    model = None
    for i in range(0, 5):
        for parameters in total_num_parameters:
            for nn_type in network_types:
                nodes_in_layer = gf.get_nodes_in_layer(parameters, nn_type)
                extra_layers = []
                for n in range(0, i):
                    extra_layers.append(gf.get_nodes_in_layer(num_divisible_by_all, nn_type))
                extra_layers.append(nodes_in_layer)
                nodes_in_layer = extra_layers
                for activation_func in activation_functions:
                    start = 5
                    prev = 0
                    smallest_not_retained = 30
                    largest_retained = 0
                    print("Thread", thread, "nodes_in_layer", nodes_in_layer, "parameters", parameters, "nn_type",
                          nn_type, "activation_func", activation_func)
                    if not gf.log_contains(log_name=logfile, nn_type=nn_type, activation_func=activation_func,
                                           parameters=parameters,
                                           nodes_in_layer=str(nodes_in_layer)):

                        while (smallest_not_retained - largest_retained) > 1:
                            x_train, y_train, x_test, y_test = generate_sets_class(start)
                            score_after_training_net, model = gf.train_test_neural_net_architecture(
                                x_train, y_train,
                                x_test, y_test,
                                nodes_in_layer=nodes_in_layer,
                                nodes_in_out_layer=start,
                                nn_type=nn_type,
                                activation_func=activation_func,
                                verbose=1,
                                one_hot=True)

                            #
                            if score_after_training_net > 0.98:
                                print("   -> ", start)
                                largest_retained = start
                                prev = start
                                start *= 2
                                if start > smallest_not_retained:
                                    start = smallest_not_retained - 1
                            else:
                                print("   <- ", start)
                                smallest_not_retained = start
                                start = int((start + prev) / 2)
                            print(" Current Num patterns", start)
                            print(" diff", str((smallest_not_retained - largest_retained)))
                            print(" smallest_not_retained", smallest_not_retained)
                            print(" largest_retained", largest_retained)
                            print(" score", score_after_training_net)

                        logging.log(logging.INFO,
                                    str(nn_type) + ";" + str(activation_func) + ";" + str(parameters) + ";" + str(
                                        nodes_in_layer) + ";" + str(largest_retained) + ";" + str(
                                        smallest_not_retained) + ";" + str(model.count_params()) + ";" + str(
                                        model.history.epoch[-1]))
                    else:
                        print("Already ran", str(nn_type), str(activation_func), str(parameters), str(nodes_in_layer))


# if __name__ == "__main__":
#     main()


# train_test_neural_net_architecture(num_patterns=10,nodes_in_layer=10, nn_type="jordan", activation_func="sigmoid")

if __name__ == "__main__":
    """
    Runs all other experiments. Pipes them to the background

    Runner is [1, 2, 3, 4, 5]
    :return:
    """

    if len(sys.argv[1:]) != 0:
        total_num_parameters = [int(x) for x in sys.argv[1:][0].split(",")]
        runner = int(sys.argv[1:][1])
        thread = int(sys.argv[1:][2])
        run_num_patterns(total_num_parameters=total_num_parameters, runner=runner, thread=thread)
