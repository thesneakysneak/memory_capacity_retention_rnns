import keras
import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score
import pandas as pd
import random

from generic_functions import EarlyStopByF1OneHot

length_of_series = 1000


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))


import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score

import random


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(np.linalg.norm(x - predicted_value)))



x=[]
y=[]
y_unscaled=[]
import random

largest_element = 5
max_count = 3

sequence_len = max_count

for i in range(500):
    k = i % max_count + 1
    print(k)
    set_of_nums = random.sample([0]*sequence_len,(sequence_len-k)) + [1]*k
    random.shuffle(set_of_nums)
    x.append([[float(i)] for i in set_of_nums])
    y.append([1/k])
    y_unscaled.append(k)

single_list = list(zip(x, y, y_unscaled))
random.shuffle(single_list)

x, y, y_unscaled = zip(*single_list)

x = numpy.asarray(x)
y = numpy.asarray(y)



inp = Input(shape=(None, 1))
ls = LSTM(1, activation="sigmoid")(inp)
output = Dense(1)(ls)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')



early_stp = EarlyStopping(monitor="val_loss", patience=20)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=1, min_lr=0.00000001)

model.fit(x, y, callbacks=[early_stp, reduce_lr], batch_size=3, epochs=100)

y_predicted = model.predict(x)

for i in range(y_predicted.shape[0]):
    p = convert_to_closest(1/y_predicted[i][0], set([x for x in range(1, max_count+1)]))
    y_e = convert_to_closest(1/y[i], set([x for x in range(1, max_count+1)]))
    print(p, y_predicted[i][0],
          y_e, y[i], (y_e == p),
          y_unscaled[i])
    # print( "     ", y_predicted[i][0], y[i], y_unscaled[i])


