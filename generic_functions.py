import random

import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, SimpleRNN, GRU, Dense
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
    print(possible_values, predicted_value)
    return min(possible_values, key=lambda x: abs(x - predicted_value))



def divisible_by_all(n):
    j = i = 0
    y = []
    while j < n:
        i += 1
        x = 12*i
        if x % 9 == 0:
            y.append(x)
            j += 1
    return y


def determine_f_score(predicted_one_hot, test_one_hot, f_only=True):
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


def get_nodes_in_layer(num_parameters, nn_type):
    if nn_type == const.LSTM:
        return int(num_parameters / 12)
    if nn_type == const.GRU:
        return int(num_parameters / 9)
    return int(num_parameters / 3)


def get_runner_experiments(runner, total_num_parameters, num_workers=5):
    splitter = int(len(total_num_parameters)/num_workers)
    total = np.array(total_num_parameters).reshape(-1, splitter)
    for i in range(splitter):
        if i % 2 == 0:
            total[i] = sorted(total[i], reverse=True)
    # Weird inconsistencies
    # total = np.transpose(total).reshape(-1, 5)
    return [total[i][runner-1] for i in range(len(total))]


def train_test_neural_net_architecture(x_train, y_train,
                                       x_test, y_test,
                                       nodes_in_layer=2, nodes_in_out_layer=1,
                                       nn_type="lstm", activation_func="sigmoid",
                                       verbose=0):

    #
    batch_size = 10
    #
    inp = Input(shape=(len(x_train[0]), 1))
    if nn_type == const.LSTM:
        ls = LSTM(nodes_in_layer, activation=activation_func)(inp)
    elif nn_type == const.ELMAN_RNN:
        ls = SimpleRNN(nodes_in_layer, activation=activation_func)(inp)
    elif nn_type == const.GRU:
        ls = GRU(nodes_in_layer, activation=activation_func)(inp)
    else:
        ls = JordanRNNCell(nodes_in_layer, activation=activation_func)(inp)
    #
    output = Dense(nodes_in_out_layer)(ls)
    #
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.05, patience=10, min_lr=0.0000001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train,
              validation_split=.2,
              callbacks=[reduce_lr, recurrent_models.earlystop2],
              epochs=1000,
              batch_size=batch_size,
              verbose=verbose)
    #
    y_predict = model.predict(x_test)
    #
    return determine_f_score(y_predict, y_test)
