
# model, result = recurrent_models.train_model(input_set=x_train, output_set=y_train, model=model,
#                                              training_alg="adam", batch_size=batch_size, use_early_stop=False,
#                                              verbose=1)
#
# model.predict(x_test, batch_size=batch_size)

# target = 100
# target_found = False
# start = 1
# prev = 1
# steps = 0
# upper_limit = 1000
# while not target_found:
#     steps += 1
#     if start < target:
#         prev = start
#         start *= 2
#         if start > upper_limit:
#             start = upper_limit - 1
#             upper_limit = start
#     elif start > target:
#         upper_limit = start
#         start = int((start+prev)/2)+1
#     elif target == start:
#         target_found = True
#     print(start)
#
#
# print(steps)

#
# random.seed(1000)
#
# inp = Input(shape=(1, 1))
# ls = LSTM(1)(inp)
# output = Dense(1)(ls)
#
#
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
# model = Model(inputs=[inp], outputs=[output])
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# model.fit(x_train, y_train, validation_split=.2, callbacks=[reduce_lr], epochs=100)





import random
import logging
import os
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense, SimpleRNN, GRU
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import recurrent_models
import ExampleRNNs.experiment_constants as const
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

def determine_score(predicted_one_hot, test_one_hot, f_only=True):
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


def true_accuracy(y_predict, y_true):
    y_true = [np.argmax(x) for x in y_true]
    y_predict_unscaled = [np.argmax(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)

def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))

def generate_sets(num_patterns):
    x = random.sample(range(1, num_patterns + 1), num_patterns)
    y = random.sample(range(1, num_patterns + 1), num_patterns)
    #
    x = [1.0 / z for z in x]
    y = [1.0 / z for z in y]
    #
    training_set = list(zip(x, y))
    training_set = training_set * 1000
    random.shuffle(training_set)
    #
    test_set = list(zip(x, y))
    test_set = test_set * 100
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
    training_set = training_set * 1000
    random.shuffle(training_set)
    #
    test_set = list(zip(x, y))
    test_set = test_set * 100
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

def train_test_neural_net_architecture(num_patterns=2, nodes_in_layer=2, nn_type="lstm", activation_func="sigmoid"):
    x_train, y_train, x_test, y_test = generate_sets_class(num_patterns)  # generate_sets(50)
    #
    batch_size = 10
    #
    inp = Input(shape=(1, 1))
    if nn_type == const.LSTM:
        ls = SimpleRNN(nodes_in_layer)(inp)
    elif nn_type == const.ELMAN_RNN:
        ls = SimpleRNN(nodes_in_layer)(inp)
    elif nn_type == const.JORDAN_RNN:
        ls = SimpleRNN(nodes_in_layer)(inp)
    else:
        ls = GRU(nodes_in_layer)(inp)
    #
    output = Dense(num_patterns)(ls)
    #
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.05, patience=10, min_lr=0.0000001)
    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, validation_split=.2, callbacks=[reduce_lr, recurrent_models.earlystop2], epochs=1000, verbose=0)
    #
    y_predict = model.predict(x_test)
    #
    return determine_score(y_predict, y_test)

random.seed(1000)
total_num_parameters = divisible_by_all(30)
activation_functions = ["softmax",
                            "elu", "selu", "softplus",
                            "softsign", "tanh", "sigmoid",
                            "hard_sigmoid",
                            "relu",
                              "linear"]
network_types = [const.LSTM, const.GRU, const.ELMAN_RNN, const.JORDAN_RNN] # "jordan_rnn"
thread = 1
run = 1
logfile_location = "danny_masters"
logfile = logfile_location + "/" +str(thread) + "_" + str(run) + "_num_patterns.log"
logfile = os.path.abspath(logfile)

if not os.path.exists(logfile):
    f = open(logfile, "a")
    f.write("")
    f.close()

logging.basicConfig(filename=logfile, level=logging.INFO)

num_patterns = 5


# possible_classes = np.unique(y_test)
# y_predict_ = [convert_to_closest(x, possible_classes) for x in y_predict]

# count = 0
# for t,p in zip(y_test_class, y_predict_class):
#     if t == p:
#         count += 1


# print("r2_score", true_accuracy(y_predict, y_test))

# d = {}
# for x, y, z in zip(y_predict_, y_test, y_predict):
#     if str(y) not in d.keys():
#         d[str(y)] = []
#     else:
#         d[str(y)].extend(z)
#
# for key in d.keys():
#     d[key] = np.unique(d[key])
#
# logging.log(logging.INFO, r2_score(y_predict_, y_test))



start = 2
prev = 1
steps = 0
smallest_not_retained = 1000
largest_retained = 0
while (smallest_not_retained - largest_retained) > 1:
    score_after_training_net = train_test_neural_net_architecture(num_patterns=start, nodes_in_layer=2, nn_type="lstm", activation_func="sigmoid")
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
        start = int((start+prev)/2)
    print(" Current Num patterns", start)
    print(" diff", str((smallest_not_retained - largest_retained)))
    print(" smallest_not_retained", smallest_not_retained)
    print(" largest_retained", largest_retained)
    print(" score", score_after_training_net)




