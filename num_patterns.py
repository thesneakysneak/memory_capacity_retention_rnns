import logging
import os
import random
import sys

import gc
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, SimpleRNN, GRU, Concatenate, LSTM
from keras import backend as K
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


def generate_sets(num_patterns, one_hot=False):
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
    x = [z / num_patterns for z in x]

    if one_hot:
        y = np.eye(num_patterns)
    else:
        y = [z / num_patterns for z in y]
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

    # Numpy does not know how to deal with tuples
    x_train = list(x_train)
    y_train = list(y_train)

    x_test = list(x_test)
    y_test = list(y_test)

    #
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 1, 1).astype(np.float32)

    if one_hot:
        y_train = y_train.astype(np.float32)
    else:
        y_train = y_train.reshape(-1, 1).astype(np.float32)
    #
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 1, 1).astype(np.float32)


    if one_hot:
        y_test = y_test.astype(np.float32)
    else:
        y_test = y_test.reshape(-1, 1).astype(np.float32)

    #
    return x_train, y_train, x_test, y_test



def run_num_patterns(total_num_parameters=[1, 2], runner=1, thread=1, one_hot=False):
    if one_hot:
        activation_functions = ["LeakyReLU", "softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid",
                                "relu",
                                "linear"]
    else:
        activation_functions = ["LeakyReLU", "elu", "selu", "tanh", "sigmoid", "hard_sigmoid", "relu", "linear"]

    network_types = [const.JORDAN_RNN, const.BIDIRECTIONAL_JORDAN_RNN,
                     const.LSTM, const.GRU, const.ELMAN_RNN,
                     const.BIDIRECTIONAL_RNN, const.BIDIRECTIONAL_LSTM,
                     const.BIDIRECTIONAL_GRU, ]  # "jordan_rnn" const.JORDAN_RNN

    run = runner
    logfile_location = "danny_masters"
    logfile = logfile_location + "/" + str(thread) + "_" + str(run) + "_" + str(one_hot)+ "_num_patterns.log"
    logfile = os.path.abspath(logfile)

    if not os.path.exists(logfile):
        f = open(logfile, "w")
        f.write(
            "nn_type;activation_func;parameters;nodes_in_layer;"
            +"largest_retained;smallest_not_retained;model_params;"
            +"num_epochs;model_score;highest_F1;optimizer\n")
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
    for additional_layers in range(0, 5):
        for parameters in total_num_parameters:
            for nn_type in network_types:
                nodes_in_layer = gf.get_nodes_in_layer(parameters, nn_type)
                extra_layers = []
                for n in range(0, additional_layers):
                    extra_layers.append(gf.get_nodes_in_layer(num_divisible_by_all, nn_type))
                extra_layers.append(nodes_in_layer)
                nodes_in_layer = extra_layers
                for activation_func in activation_functions:
                    for optimizer in const.LIST_OF_OPTIMIZERS:
                        start = 5
                        prev = 0
                        smallest_not_retained = 30
                        largest_retained = 0
                        print("Thread", thread, "nodes_in_layer", nodes_in_layer, "parameters", parameters, "nn_type",
                              nn_type, "activation_func", activation_func, "optimizer", str(optimizer))
                        if not gf.log_contains(log_name=logfile, nn_type=nn_type, activation_func=activation_func,
                                               parameters=parameters,
                                               nodes_in_layer=str(nodes_in_layer),
                                               optimizer=str(optimizer)):

                            while (smallest_not_retained - largest_retained) > 1:
                                x_train, y_train, x_test, y_test = generate_sets(start, one_hot=one_hot)
                                score_after_training_net, model = gf.train_test_neural_net_architecture(
                                    x_train, y_train,
                                    x_test, y_test,
                                    nodes_in_layer=nodes_in_layer,
                                    nodes_in_out_layer=y_test.shape[1],
                                    nn_type=nn_type,
                                    activation_func=activation_func,
                                    optimizer=optimizer,
                                    verbose=0,
                                    one_hot=one_hot)

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
                                                model.history.epoch[-1]) + ";" + str("") +
                                            ";"+str(score_after_training_net) + ";" + str(optimizer))

                                K.clear_session()
                                gc.collect()
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
        if sys.argv[1:][3] == "False":
            one_hot = False
        else:
            one_hot = True
        print("one_hot", one_hot)
        run_num_patterns(total_num_parameters=total_num_parameters,
                         runner=runner, thread=thread,
                         one_hot=one_hot)
