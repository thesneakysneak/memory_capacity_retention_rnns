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


def generate_sets(sequence_length=1000, num_samples=10):
    x = x_test = [0] * 5000
    y = y_test = [0] * 5000

    for i in range(5000):
        num_samples = 5000
        k = random.randint(5, 45)
        set_of_nums = random.sample([1, 2] * 10000, (sequence_length - k)) + [3] * k
        random.shuffle(set_of_nums)
        x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
        y[i] = numpy.array(1. / k).astype(np.float32)

    x = numpy.array(x)
    y = numpy.array(y)
    return x, y, x_test, y_test

x_train, y_train, x_test, y_test = generate_sets(100, 10)

inp = Input(shape=(5000, 1))
ls = SimpleRNN(1)(inp)
output = Dense(1)(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, validation_split=.2, callbacks=[reduce_lr])
