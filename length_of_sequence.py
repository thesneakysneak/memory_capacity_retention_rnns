import sys

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

length_of_series = 1000


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))


def generate_count_set(sequence_length_=3000, max_count=10, total_num_patterns=100):
    x = [0] * total_num_patterns
    y = [0] * total_num_patterns
    #
    for i in range(total_num_patterns):
        k = i % max_count + 1
        set_of_nums = random.sample([1, 2] * sequence_length_, (sequence_length_ - k)) + [3] * k
        random.shuffle(set_of_nums)
        x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
        y[i] = numpy.array(1. / k).astype(np.float32)
    #
    single_list = list(zip(x, y))
    random.shuffle(single_list)
    x, y = zip(*single_list)
    #
    x = numpy.array(x)
    y = numpy.array(y)
    return x, y


def run_experiment(max_count=2, nodes_in_layer=2, nn_type="lstm", activation_func="sigmoid", verbose=0):
    sequence_length = 1000
    x_train, y_train = generate_count_set(sequence_length_=sequence_length,
                                          max_count=max_count,
                                          total_num_patterns=300)  # generate_sets(50)
    x_test, y_test = generate_count_set(sequence_length_=sequence_length,
                                        max_count=max_count,
                                        total_num_patterns=300)


    result = gf.train_test_neural_net_architecture(x_train, y_train,
                                       x_test, y_test,
                                       nodes_in_layer=nodes_in_layer,
                                       nodes_in_out_layer=1,
                                       nn_type=nn_type, activation_func=activation_func,
                                       verbose=1)

    return result



def run_length_experiment(total_num_parameters=[1, 2], runner=1, thread=1):
    activation_functions = ["softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "relu",
                            "linear"]
    network_types = [const.LSTM, const.GRU, const.ELMAN_RNN, const.JORDAN_RNN]  # "jordan_rnn"

    logfile_location = "danny_masters"
    logfile = logfile_location + "/" + str(thread) + "_" + str(runner) + "_longest_sequence.log"
    logfile = os.path.abspath(logfile)

    if not os.path.exists(logfile):
        f = open(logfile, "a")
        f.write("")
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

    for parameters in total_num_parameters:
        for nn_type in network_types:
            nodes_in_layer = gf.get_nodes_in_layer(parameters, nn_type)
            for activation_func in activation_functions:
                print("Thread", thread, "parameters", parameters, "nn_type", nn_type, "activation_func", activation_func)
                while (smallest_not_retained - largest_retained) > 1:
                    score_after_training_net = run_experiment(max_count=start,
                                                              nodes_in_layer=nodes_in_layer,
                                                              nn_type=nn_type,
                                                              activation_func=activation_func)
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

                logging.log(logging.INFO, str(nn_type) + "," + str(activation_func) + "," + str(parameters) + "," + str(
                    nodes_in_layer) + "," + str(largest_retained) + "," + str(smallest_not_retained))


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
        run_length_experiment(total_num_parameters=total_num_parameters, runner=runner, thread=thread)
    else:
        sample()