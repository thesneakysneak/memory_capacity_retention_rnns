import os
import random

import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Bidirectional
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support

import experiment_constants as const
import recurrent_models
from scratch_space.jordan_rnn import JordanRNNCell


def get_runner_experiments(runner, total_num_parameters):
    total_num_parameters = np.array(total_num_parameters).reshape(-1, 5)
    for i in range(5):
        if i % 2 == 0:
            total_num_parameters[i] = sorted(total_num_parameters[i], reverse=True)
    total_num_parameters = np.transpose(total_num_parameters)
    random.shuffle(total_num_parameters)
    return total_num_parameters[runner]


def true_accuracy_one_hot(y_predict, y_true):
    y_true = [np.argmax(x) for x in y_true]
    y_predict_unscaled = [np.argmax(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    # print(possible_values, predicted_value)
    return min(possible_values, key=lambda x: abs(x - predicted_value))


def divisible_by_all(n):
    j = i = 0
    y = []
    while j < n:
        i += 1
        x = 24 * i
        if x % 9 == 0 and x % 12 == 0:
            y.append(x)
            j += 1
    return y

def determine_f_score_one_hot(predicted_one_hot, test_one_hot, f_only=True):
    """
    :param predicted_one_hot: Output produced by the network which is one hot encoded
    :param test_one_hot: Output expected which is one hot encoded
    :param f_only: Boolean flag indicating whether only the score is required
    :return: performance of the network, which is either only the fscore or
                                        precision, recall, fbeta_score, beta
    """
    p_categories = [np.argmax(x) for x in predicted_one_hot]
    t_categories = [np.argmax(x) for x in test_one_hot]
    # for i in range(len(p_categories)):
    #     if p_categories[i] != t_categories[i]:
    #         print("Class not correct", p_categories[i], t_categories[i])
    conf_mat = confusion_matrix(t_categories, p_categories)
    precision, recall, fbeta_score, beta = precision_recall_fscore_support(t_categories, p_categories, average="micro")
    # print(conf_mat)
    if f_only:
        return fbeta_score
    return precision, recall, fbeta_score, conf_mat

def determine_f_score(predicted, test, f_only=True):
    """
    :param predicted_one_hot: Output produced by the network which is one hot encoded
    :param test_one_hot: Output expected which is one hot encoded
    :param f_only: Boolean flag indicating whether only the score is required
    :return: performance of the network, which is either only the fscore or
                                        precision, recall, fbeta_score, beta
    """
    p_categories = [int(1/x) if x > 0.0000001 else 0 for x in predicted ]
    t_categories = [int(1/x) if x > 0.0000001 else 0 for x in test]
    print(p_categories)
    print(t_categories)

    p_categories = [convert_to_closest(x, list(set(t_categories))) for x in p_categories]

    # for i in range(len(p_categories)):
    #     if p_categories[i] != t_categories[i]:
    #         print("Class not correct", p_categories[i], t_categories[i])
    conf_mat = confusion_matrix(t_categories, p_categories)
    precision, recall, fbeta_score, beta = precision_recall_fscore_support(t_categories, p_categories, average="micro")
    # print(conf_mat)
    if f_only:
        return fbeta_score
    return precision, recall, fbeta_score, conf_mat

def determine_ave_f_score(predicted, test, f_only=True):
    fscore = 0.0
    for i in range(len(predicted)):
        fscore += determine_f_score(predicted[i], test[i])
    return fscore/len(predicted)

def get_nodes_in_layer(num_parameters, nn_type):
    if nn_type == const.BIDIRECTIONAL_LSTM:
        return int(num_parameters / 24)
    if nn_type == const.LSTM:
        return int(num_parameters / 12)
    if nn_type == const.GRU:
        return int(num_parameters / 9)
    return int(num_parameters / 3)


def get_runner_experiments(runner, total_num_parameters, num_workers=5):
    splitter = int(len(total_num_parameters) / num_workers)
    total = np.array(total_num_parameters).reshape(-1, splitter)
    return total[runner - 1]


def are_sets_correlated(set1, set2):
    from scipy.stats import spearmanr
    coef, p = spearmanr(set1, set2)
    print('Spearmans correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.05
    if p > alpha or (coef <= 0.1 and coef >= -1):
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p, coef)
        return False
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
        return True

def train_test_neural_net_architecture(x_train, y_train,
                                       x_test, y_test,
                                       nodes_in_layer=2, nodes_in_out_layer=1,
                                       nn_type="lstm", activation_func="sigmoid",
                                       verbose=0):
    #
    batch_size = int(len(x_train)*0.10)
    #
    inp = Input(shape=(len(x_train[0]), 1))
    if type(nodes_in_layer) == int:
        if nn_type == const.LSTM:
            ls = LSTM(nodes_in_layer, activation=activation_func)(inp)
        elif nn_type == const.ELMAN_RNN:
            ls = SimpleRNN(nodes_in_layer, activation=activation_func)(inp)
        elif nn_type == const.GRU:
            ls = GRU(nodes_in_layer, activation=activation_func)(inp)
        elif nn_type == const.BIDIRECTIONAL_LSTM:
            ls = Bidirectional(LSTM(nodes_in_layer, activation=activation_func))(inp)
    elif type(nodes_in_layer) == list:
        rnn_func_ptr = None
        if nn_type == const.LSTM:
            rnn_func_ptr = LSTM
        elif nn_type == const.ELMAN_RNN:
            rnn_func_ptr = SimpleRNN
        elif nn_type == const.GRU:
            rnn_func_ptr = GRU
        else:
            # TODO Figure this out
            rnn_func_ptr = SimpleRNN

        if len(nodes_in_layer) > 1:
            ls = rnn_func_ptr(nodes_in_layer[0], activation=activation_func, return_sequences=True)(inp)
            for n in range(1, len(nodes_in_layer) - 1):
                ls = rnn_func_ptr(nodes_in_layer[n], activation=activation_func, return_sequences=True)(ls)
            ls = rnn_func_ptr(nodes_in_layer[-1], activation=activation_func)(ls)
        else:
            ls = rnn_func_ptr(nodes_in_layer[0], activation=activation_func)(inp)

    #
    output = Dense(nodes_in_out_layer)(ls)
    #
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.05, patience=10, min_lr=0.0000001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train,
              validation_split=.2,
              callbacks=[reduce_lr, recurrent_models.earlystop],
              epochs=1000,
              batch_size=batch_size,
              verbose=verbose)
    #
    y_predict = model.predict(x_test)
    #

    if len(y_predict[0]) > 1:
        print(y_predict)
        return determine_ave_f_score(y_predict, y_test), model
    return determine_f_score(y_predict, y_test), model

def simple_bidirectional():
    import os
    import random

    import numpy as np
    from keras import Input, Model
    from keras.callbacks import ReduceLROnPlateau
    from keras.layers import LSTM, SimpleRNN, GRU, Dense, Bidirectional
    from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support

    import experiment_constants as const
    import recurrent_models

    nodes_in_out_layer = 1
    nodes_in_layer = 1
    activation_func="sigmoid"
    batch_size = 10

    inp = Input(shape=(1, 1))
    ls = Bidirectional(LSTM(nodes_in_layer, activation=activation_func))(inp)
    output = Dense(nodes_in_out_layer)(ls)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.05, patience=10, min_lr=0.0000001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(loss='mse', optimizer='adam')


    model.fit(x, y, epochs=2200, verbose=2, callbacks=[reduce_lr])


def log_contains(log_name, nn_type, activation_func, parameters, nodes_in_layer):
    # TODO
    if not os.path.exists(log_name):
        return False
    import pandas as pd
    df = pd.read_csv(log_name, delimiter=";")
    df_found =   df[(df["nn_type"] == nn_type) & (df["activation_func"] == activation_func) & (df["parameters"] == parameters) & (
            df["nodes_in_layer"] == nodes_in_layer)]
    if df_found.empty:
        return False
    return True

def set_check_point(thread, runner, experiment, parameters, architecture, neural_network_type):
    check_point_file = "danny_masters"
    check_point_file = check_point_file + "/" + str(thread) + "_" + str(runner) + "_" + experiment + ".txt"
    check_point_file = os.path.abspath(check_point_file)
    line = str(parameters) + ";" + str(architecture) + ";" + str(neural_network_type)
    f = open(check_point_file, "a")
    f.write(line)
    f.close()

def load_check_point(thread, runner, experiment):
    check_point_file = "danny_masters"
    check_point_file = check_point_file + "/" + str(thread) + "_" + str(runner) + "_" + experiment + ".txt"
    check_point_file = os.path.abspath(check_point_file)

    f = open(check_point_file, "r")
    checkpoint = []
    line = f.readline()
    while line:
        checkpoint.append(line)
        line = f.readline()
    f.close()

    checkpoint = checkpoint[0].split(";")
    parameters = eval(checkpoint[0])
    architecture = eval(checkpoint[1])
    neural_network_type = checkpoint[2]

    return parameters, architecture, neural_network_type

