import os
import random

import keras
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Bidirectional
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support

import experiment_constants as const
import recurrent_models
from scratch_space.jordan_rnn import JordanRNNCell



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
            score = determine_ave_f_score(predict, target)
        else:
            score = determine_f_score(predict, target)
        self.delta_score = score - self.prev_delta_score
        self.prev_delta_score = score

        print("Epoch %05d: delta_score" % epoch, score, self.delta_score, self.patience)
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
        score = determine_f_score(y_predict_unscaled, y_true)
        self.delta_score = score - self.prev_delta_score
        self.prev_delta_score = score

        print("Epoch %05d: delta_score" % epoch, score, self.delta_score, self.patience)
        if np.abs(self.delta_score) < 0.05:
            self.patience += 1
        else:
            self.patience = 0

        if self.patience >= 700 or score > 0.98:
            if self.verbose > 0:
                print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True























###########################################################
#
#######################################################
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
    return min(possible_values, key=lambda x: abs(np.linalg.norm(x - predicted_value)))



def divisible_by_all(n):
    j = i = 0
    y = []
    while j < n:
        i += 1
        x = 24 * i
        if x % 9 == 0 and x % 12 == 0 and x % 21 == 0:
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
    # print(p_categories)
    # print(t_categories)

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
    if nn_type == const.BIDIRECTIONAL_RNN:
        return int(num_parameters / 6)
    if nn_type == const.BIDIRECTIONAL_GRU:
        return int(num_parameters / 21)
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
                                       verbose=0, epocs=10000,
                                       one_hot=False):
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
        elif nn_type == const.BIDIRECTIONAL_GRU:
            ls = Bidirectional(GRU(nodes_in_layer, activation=activation_func))(inp)
        elif nn_type == const.BIDIRECTIONAL_RNN:
            ls = Bidirectional(SimpleRNN(nodes_in_layer, activation=activation_func))(inp)
    elif type(nodes_in_layer) == list:
        if len(nodes_in_layer) > 1:
            in_layer = inp
            if nn_type == const.LSTM:
                ls = LSTM(nodes_in_layer[0], activation=activation_func, return_sequences=True)(inp)
            elif nn_type == const.ELMAN_RNN:
                ls = SimpleRNN(nodes_in_layer[0], activation=activation_func, return_sequences=True)(inp)
            elif nn_type == const.GRU:
                ls = GRU(nodes_in_layer[0], activation=activation_func, return_sequences=True)(inp)
            elif nn_type == const.BIDIRECTIONAL_LSTM:
                ls = Bidirectional(LSTM(nodes_in_layer[0], activation=activation_func, return_sequences=True))(inp)
            elif nn_type == const.BIDIRECTIONAL_GRU:
                ls = Bidirectional(GRU(nodes_in_layer[0], activation=activation_func, return_sequences=True))(inp)
            elif nn_type == const.BIDIRECTIONAL_RNN:
                ls = Bidirectional(SimpleRNN(nodes_in_layer[0], activation=activation_func, return_sequences=True))(inp)


            for n in range(1, len(nodes_in_layer) - 1):
                if nn_type == const.LSTM:
                    ls = LSTM(nodes_in_layer[0], activation=activation_func, return_sequences=True)(ls)
                elif nn_type == const.ELMAN_RNN:
                    ls = SimpleRNN(nodes_in_layer[0], activation=activation_func, return_sequences=True)(ls)
                elif nn_type == const.GRU:
                    ls = GRU(nodes_in_layer[0], activation=activation_func, return_sequences=True)(ls)
                elif nn_type == const.BIDIRECTIONAL_LSTM:
                    ls = Bidirectional(LSTM(nodes_in_layer[0], activation=activation_func, return_sequences=True))(ls)
                elif nn_type == const.BIDIRECTIONAL_GRU:
                    ls = Bidirectional(GRU(nodes_in_layer[0], activation=activation_func, return_sequences=True))(ls)
                elif nn_type == const.BIDIRECTIONAL_RNN:
                    ls = Bidirectional(SimpleRNN(nodes_in_layer[0], activation=activation_func, return_sequences=True))(ls)

            if nn_type == const.LSTM:
                ls = LSTM(nodes_in_layer[-1], activation=activation_func)(ls)
            elif nn_type == const.ELMAN_RNN:
                ls = SimpleRNN(nodes_in_layer[-1], activation=activation_func)(ls)
            elif nn_type == const.GRU:
                ls = GRU(nodes_in_layer[-1], activation=activation_func)(ls)
            elif nn_type == const.BIDIRECTIONAL_LSTM:
                ls = Bidirectional(LSTM(nodes_in_layer[-1], activation=activation_func))(ls)
            elif nn_type == const.BIDIRECTIONAL_GRU:
                ls = Bidirectional(GRU(nodes_in_layer[-1], activation=activation_func))(ls)
            elif nn_type == const.BIDIRECTIONAL_RNN:
                ls = Bidirectional(SimpleRNN(nodes_in_layer[-1], activation=activation_func))(ls)

        else:
            if nn_type == const.LSTM:
                ls = LSTM(nodes_in_layer[0], activation=activation_func)(inp)
            elif nn_type == const.ELMAN_RNN:
                ls = SimpleRNN(nodes_in_layer[0], activation=activation_func)(inp)
            elif nn_type == const.GRU:
                ls = GRU(nodes_in_layer[0], activation=activation_func)(inp)
            elif nn_type == const.BIDIRECTIONAL_LSTM:
                ls = Bidirectional(LSTM(nodes_in_layer[0], activation=activation_func))(inp)
            elif nn_type == const.BIDIRECTIONAL_GRU:
                ls = Bidirectional(GRU(nodes_in_layer[0], activation=activation_func))(inp)
            elif nn_type == const.BIDIRECTIONAL_RNN:
                ls = Bidirectional(SimpleRNN(nodes_in_layer[0], activation=activation_func))(inp)

                #
    output = Dense(nodes_in_out_layer)(ls)
    #
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)

    if one_hot:
        earlystop = EarlyStopByF1OneHot(value=.95, verbose=1)  # EarlyStopByF1(value=.99, verbose=1)
    else:
        earlystop = EarlyStopByF1(value=.95, verbose=1)
        # earlystop = EarlyStopping(monitor='val_loss',
        #                            min_delta=0,
        #                            patience=200,
        #                            verbose=0, mode='auto')


    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train,
              validation_split=.2,
              callbacks=[reduce_lr, earlystop],
              epochs=epocs,
              batch_size=batch_size,
              verbose=verbose)
    #
    y_predict = model.predict(x_test)
    #

    if len(y_predict[0]) > 1:
        print(y_predict)
        return determine_ave_f_score(y_predict, y_test), model
    return determine_f_score(y_predict, y_test), model

def test_train_nn():
    x_train =[random.random() for i in range(100)]
    y_train = [random.random() for i in range(100)]

    x_train= np.array(x_train).reshape(-1, 1, 1).astype(np.float32)
    y_train = np.array(y_train).reshape(-1, 1).astype(np.float32)


    x_test = x_train
    y_test = y_train

    activation_functions = ["softmax", "elu", "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "relu",
                            "linear"]
    network_types = [const.LSTM, const.GRU, const.ELMAN_RNN, const.BIDIRECTIONAL_RNN, const.BIDIRECTIONAL_LSTM, const.BIDIRECTIONAL_GRU]  # "jordan_rnn" const.JORDAN_RNN

    for a in activation_functions:
        for n in network_types:
            print("             ", n)
            score, model = train_test_neural_net_architecture(x_train, y_train,
                                               x_test, y_test,
                                               nodes_in_layer=2, nodes_in_out_layer=1,
                                               nn_type=n, activation_func=a,
                                               verbose=0, epocs=2)
            score, model = train_test_neural_net_architecture(x_train, y_train,
                                                              x_test, y_test,
                                                              nodes_in_layer=[2], nodes_in_out_layer=1,
                                                              nn_type=n, activation_func=a,
                                                              verbose=0, epocs=2)

            score, model = train_test_neural_net_architecture(x_train, y_train,
                                                              x_test, y_test,
                                                              nodes_in_layer=[2, 2, 2, 2], nodes_in_out_layer=1,
                                                              nn_type=n, activation_func=a,
                                                              verbose=0, epocs=2)
            print(model.summary())

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

    x = [random.random() for i in range(100)]
    y = [random.random() for i in range(100)]
    # x = y
    x = np.array(x).reshape(-1, 1, 1).astype(np.float32)
    y = np.array(y).reshape(-1, 1).astype(np.float32)

    inp = Input(shape=(None, len(x[0])))
    ls = Bidirectional(SimpleRNN(nodes_in_layer, activation=activation_func))(inp)
    output = Dense(nodes_in_out_layer)(ls)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0000001)
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

