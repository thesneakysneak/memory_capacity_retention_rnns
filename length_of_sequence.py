import sys
from keras import backend as K
import numpy
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score

import logging
import os
import random

import experiment_constants as const
import recurrent_models
import generic_functions as gf
import pandas as pd
import tensorflow as tf

length_of_series = 1000


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))


def generate_count_set(sequence_length_=100, max_count=10, total_num_patterns=100, one_hot=False):
    x = []
    y = []
    y_unscaled = []


    for i in range(max_count):
        k = i % max_count + 1
        print(k)
        set_of_nums = random.sample([0] * sequence_length_, (sequence_length_ - k)) + [1] * k
        random.shuffle(set_of_nums)
        x.append([[i] for i in set_of_nums])
        y.append([1 / k])
        y_unscaled.append(k)

    if one_hot:
        training_set = list(zip(x, np.asarray(pd.get_dummies(y_unscaled)))) * total_num_patterns
        test_set = list(zip(x, np.asarray(pd.get_dummies(y_unscaled)))) * int(total_num_patterns/10)
    else:
        training_set = list(zip(x, y)) * total_num_patterns
        test_set = list(zip(x, y)) * int(total_num_patterns/10)

    random.shuffle(training_set)
    random.shuffle(test_set)

    x_train, y_train = zip(*training_set)
    x_test, y_test = zip(*test_set)

    x_train = list(x_train)
    y_train = list(y_train)
    x_train = numpy.asarray(x_train)
    y_train = numpy.asarray(y_train)

    x_test = list(x_test)
    y_test = list(y_test)
    x_test = numpy.asarray(x_test)
    y_test = numpy.asarray(y_test)

    return x_train, y_train, x_test, y_test


def run_experiment(max_count=2, nodes_in_layer=2, nn_type="lstm", activation_func="sigmoid", verbose=0, one_hot=False):
    sequence_length = 1000
    x_train, y_train, x_test, y_test  = generate_count_set(sequence_length_=sequence_length,
                                          max_count=max_count,
                                          total_num_patterns=1000,
                                          one_hot=one_hot)  # generate_sets(50)


    result = gf.train_test_neural_net_architecture(x_train, y_train,
                                                   x_test, y_test,
                                                   nodes_in_layer=nodes_in_layer,
                                                   nodes_in_out_layer=1,
                                                   nn_type=nn_type,
                                                   activation_func=activation_func,
                                                   verbose=1)

    return result


def run_length_experiment(total_num_parameters=[1, 2], runner=1, thread=1, one_hot=False):
    activation_functions = ["softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "relu",
                            "linear"]
    network_types = [
                        const.JORDAN_RNN, const.BIDIRECTIONAL_JORDAN_RNN,
                        const.LSTM, const.GRU, const.ELMAN_RNN,
                     const.BIDIRECTIONAL_RNN, const.BIDIRECTIONAL_LSTM, const.BIDIRECTIONAL_GRU
                     ]  # "jordan_rnn" const.JORDAN_RNN

    logfile_location = "danny_masters"
    logfile = logfile_location + "/" + str(thread) + "_" + str(runner) + "_" + str(one_hot) + "_longest_sequence.log"
    logfile = os.path.abspath(logfile)

    if not os.path.exists(logfile):
        f = open(logfile, "w")
        f.write("nn_type;activation_func;parameters;nodes_in_layer;"+
                "largest_retained;smallest_not_retained;model_params;"+
                "num_epochs;model_score;highest_F1\n")
        f.close()

    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

    start = 2
    prev = 1
    steps = 0
    smallest_not_retained = 100000
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
                    start = 1
                    prev = 0
                    smallest_not_retained = 30
                    largest_retained = 0
                    print("Thread", thread, "parameters", parameters, "nn_type", nn_type, "activation_func",
                          activation_func)
                    if not gf.log_contains(log_name=logfile, nn_type=nn_type, activation_func=activation_func,
                                           parameters=parameters,
                                           nodes_in_layer=str(nodes_in_layer)):
                        while (smallest_not_retained - largest_retained) > 1:
                            x_train, y_train, x_test, y_test = generate_count_set(sequence_length_=50,
                                                                                  max_count=start,
                                                                                  total_num_patterns=100,
                                                                                  one_hot=one_hot)
                            score_after_training_net, model = gf.train_test_neural_net_architecture(
                                x_train, y_train,
                                x_test, y_test,
                                nodes_in_layer=nodes_in_layer,
                                nodes_in_out_layer=y_test.shape[1],
                                nn_type=nn_type,
                                activation_func=activation_func,
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
                                            model.history.epoch[-1])+ ";" + str("") +";"+str(score_after_training_net))
                            K.clear_session()
                    else:
                        print("Already ran", str(nn_type), str(activation_func), str(parameters), str(nodes_in_layer))


def sample():
    x, y = generate_count_set(sequence_length_=300, max_count=10, total_num_patterns=100)
    inp = Input(shape=(300, 1))
    ls = SimpleRNN(1)(inp)
    output = Dense(1)(ls)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x, y, validation_split=.2, callbacks=[reduce_lr, recurrent_models.earlystop2], epochs=100)


# if __name__ == "__main__":
#     main()


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
        one_hot = False
        if sys.argv[1:][3] == "False":
            one_hot = False
        else:
            one_hot = True
        run_length_experiment(total_num_parameters=total_num_parameters, runner=runner, thread=thread, one_hot=one_hot)
    else:
        sample()
