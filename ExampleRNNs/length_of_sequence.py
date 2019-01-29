import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score

import random

length_of_series = 1000


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))


def generate_sets(sequence_length, num_samples):
    x_train = x_test = [0] * sequence_length
    y_train = y_test = [0] * sequence_length
    #
    for i in range(1, num_samples):
        k = i + 1
        set_of_nums = random.sample([1, 2] * sequence_length, (sequence_length - k)) + [3] * k
        random.shuffle(set_of_nums)
        x_train[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
        y_train[i] = numpy.array(1. / k).astype(np.float32)
        #
        random.shuffle(set_of_nums)
        x_test[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
        y_test[i] = numpy.array(1. / k).astype(np.float32)
    #
    return x_train, y_train, x_test, y_test


x = [0] * 5000
y = [0] * 5000
import random


for i in range(5000):
    len = 5000
    k = random.randint(5, 45)
    set_of_nums = random.sample([1, 2] * 10000, (len - k)) + [3] * k
    random.shuffle(set_of_nums)
    x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
    y[i] = numpy.array(1. / k).astype(np.float32)


x = numpy.array(x)
y = numpy.array(y)

inp = Input(shape=(5000, 1))
ls = SimpleRNN(1)(inp)
output = Dense(1)(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, validation_split=.2, callbacks=[reduce_lr])
