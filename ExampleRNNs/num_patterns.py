import random
import logging
import os
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score
import recurrent_models

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


def true_accuracy(y_predict, y_true):
    y_true = [np.round(1/x) for x in y_true]
    y_predict_unscaled = [np.round(1/x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


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


total_num_parameters = divisible_by_all(30)
activation_functions = ["softmax",
                            "elu", "selu", "softplus",
                            "softsign", "tanh", "sigmoid",
                            "hard_sigmoid",
                            "relu",
                              "linear"]
network_types = ["lstm", "gru", "elman_rnn", ] # "jordan_rnn"
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


x_train, y_train, x_test, y_test = generate_sets(2)

model = recurrent_models.get_model(architecture=[1, 1, 1],
              batch_size=10,
              timesteps=1,
              network_type="lstm",
              activation_function='tanh')

result, model = recurrent_models.train_model(input_set=x_train, output_set=y_train, model=model, training_alg="adam", batch_size=10)


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



