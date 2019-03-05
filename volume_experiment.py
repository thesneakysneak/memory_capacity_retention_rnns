
# https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/
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

from keras import backend as K

def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)

def generate_volume_set(sequence_length_=3000, max_count=10, total_num_patterns=100, total_num_to_count=10):
    x = []
    y = []
    # TODO Validate this
    assert max_count*total_num_to_count < sequence_length_
    for i in range(total_num_patterns):
        random_lengths = [random.randint(1, max_count) for p in range(total_num_to_count)]
        k = sum(random_lengths)
        array_to_add = []
        for l in range(total_num_to_count):
            array_to_add.extend([(l+3)/max_count]*random_lengths[l])

        set_of_nums = random.sample([1.0/max_count] * sequence_length_,
                                    (sequence_length_ - k)) + array_to_add
        random.shuffle(set_of_nums)
        x.append(numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32))
        y.append(numpy.array([1. / p for p in random_lengths]).astype(np.float32))
    #
    single_list = list(zip(x, y))
    random.shuffle(single_list)
    x, y = zip(*single_list)

    x = list(x)
    y = list(y)

    #
    x = numpy.array(x)
    y = numpy.array(y)
    return x, y


def run_experiment(max_count=2, max_elements_to_count=2, nodes_in_layer=2, nn_type="lstm", activation_func="sigmoid", verbose=0):
    sequence_length = 3000
    x_train, y_train = generate_volume_set(sequence_length_=sequence_length,
                                           max_count=max_count,
                                           total_num_patterns=100,
                                           total_num_to_count=max_elements_to_count)  # generate_sets(50)
    x_test, y_test = generate_volume_set(sequence_length_=sequence_length,
                                         max_count=max_count,
                                         total_num_patterns=100,
                                         total_num_to_count=max_elements_to_count)

    result = gf.train_test_neural_net_architecture(x_train, y_train,
                                       x_test, y_test,
                                       nodes_in_layer=nodes_in_layer, nodes_in_out_layer=max_elements_to_count,
                                       nn_type=nn_type, activation_func=activation_func,
                                       verbose=1)

    return result


def search_in_range(nodes_in_layer, parameters, nn_type, activation_func, max_elements_to_count=1):
    """
    Function searches for the maximum length that can be counted for the given number of elements that need to be counted
    :param parameters:
    :param nn_type:
    :param activation_func:
    :param max_elements_to_count:
    :return:
    """
    start = 2
    prev = 1
    steps = 0
    score_after_training_net = 0.0
    smallest_not_retained = 30
    largest_retained = 0
    model = None
    while (smallest_not_retained - largest_retained) > 1:
        score_after_training_net, model = run_experiment(max_count=start,
                                                  max_elements_to_count=max_elements_to_count,
                                                  nodes_in_layer=nodes_in_layer,
                                                  nn_type=nn_type,
                                                  activation_func=activation_func)
        print("Score", score_after_training_net)
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
        nodes_in_layer) + "," + str(max_elements_to_count) + "," + str(max_elements_to_count)
                + str(largest_retained) + "," + str(smallest_not_retained) + ",processing")
    return score_after_training_net, largest_retained, smallest_not_retained, model


def run_volume_experiment(total_num_parameters=[], runner=1, thread=1):
    activation_functions = ["softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "relu",
                            "linear"]
    network_types = [const.LSTM, const.GRU, const.ELMAN_RNN, const.BIDIRECTIONAL_RNN, const.BIDIRECTIONAL_LSTM,
                     const.BIDIRECTIONAL_GRU]  # "jordan_rnn" const.JORDAN_RNN

    logfile_location = "danny_masters"
    logfile = logfile_location + "/" + str(thread) + "_" + str(runner) + "_volume_experiment.log"
    logfile = os.path.abspath(logfile)
    print(logfile)
    if not os.path.exists(logfile):
        f = open(logfile, "w")
        f.write("nn_type;activation_func;parameters;nodes_in_layer;largest_retained;smallest_not_retained;"
                                    + ";largest_len_retained;smallest_len_not_retained;status;model_params;num_epochs\n")
        f.close()

    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

    start = 2
    prev = 1
    steps = 0
    smallest_not_retained = 3000
    smallest_len_not_retained = 0
    largest_len_retained = 0
    max_elements_to_count = 10
    largest_retained = 0
    nodes_in_layer = 2
    activation_func = "sigmoid"
    nn_type = "lstm"
    layer_depth = 4

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
                    smallest_not_retained = 10
                    smallest_len_not_retained = 0
                    largest_len_retained = 0
                    largest_retained = 0
                    print("Thread", thread, "parameters", parameters, "nn_type", nn_type, "activation_func", activation_func)
                    if not gf.log_contains(log_name=logfile, nn_type=nn_type, activation_func=activation_func,
                                           parameters=parameters,
                                           nodes_in_layer=str(nodes_in_layer)):
                        while (smallest_not_retained - largest_retained) > 1:
                            score_after_training_net, largest_len_retained, smallest_len_not_retained, model =  search_in_range(   nodes_in_layer=nodes_in_layer,
                                                                                                                                parameters=parameters,
                                                                                                                                    nn_type=nn_type,
                                                                                                                                    activation_func=activation_func,
                                                                                                                                    max_elements_to_count=start,)

                            print(" ==================================== ", score_after_training_net)
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
                            print(" max elements to count smallest_not_retained", smallest_not_retained)
                            print(" max elements to count largest_retained", largest_retained)
                            print(" length to count smallest_not_retained", smallest_len_not_retained)
                            print(" length to count largest_retained", largest_len_retained)
                            print(" score", score_after_training_net)

                        logging.log(logging.INFO, str(nn_type) + ";" + str(activation_func) + ";" + str(parameters) + ";" + str(
                            nodes_in_layer) + ";" + str(largest_retained) + ";" + str(smallest_not_retained)
                                            + ";" + str(largest_len_retained) + ";" + str(smallest_len_not_retained)+ ";found;" \
                                    + str(model) + ";" +str(model.history.epoch[-1]))
                        K.clear_session()
                    else:
                        print("Already ran", str(nn_type), str(activation_func),str(parameters), str(nodes_in_layer))

def sample():
    x, y = generate_volume_set(sequence_length_=300, max_count=20, total_num_patterns=100, total_num_to_count=10)
    inp = Input(shape=(5000, 1))
    ls = SimpleRNN(2)(inp)
    output = Dense(2)(ls)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x, y, validation_split=.2, callbacks=[reduce_lr, recurrent_models.earlystop2], epochs=1000)

    y_predict = 1 / model.predict(x)
    y_predict = [[round(p[0]), round(p[1])] for p in y_predict]


def main():
    total_num_parameters = gf.divisible_by_all(1)
    run_volume_experiment(total_num_parameters=total_num_parameters, runner=1, thread=1)


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
        run_volume_experiment(total_num_parameters=total_num_parameters, runner=runner, thread=thread)





