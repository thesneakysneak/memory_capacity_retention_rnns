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

for i in range(100):
    k = i % max_count + 1
    print(k)
    set_of_nums = random.sample([0]*sequence_len,(sequence_len-k)) + [1]*k
    random.shuffle(set_of_nums)
    x.append([[i] for i in set_of_nums])
    y.append([1/k])
    y_unscaled.append(k)

one_hot = True
if one_hot:
    training_set = list(zip(x, np.asarray(pd.get_dummies(y_unscaled))))*100
    test_set = list(zip(x, np.asarray(pd.get_dummies(y_unscaled))))*10
else:
    training_set = list(zip(x, y))*10
    test_set = list(zip(x, y))

random.shuffle(training_set)
random.shuffle(test_set)

x_train, y_train = zip(*training_set)
x_test, y_test = zip(*test_set)


x_train = list(x_train)
y_train = list(y_train)
x_train = numpy.asarray(x_train)
y_train = numpy.asarray(y_train)


x_test = list(x_test)
y_test = list(y_test)
x_test = numpy.asarray(x_test)
y_test = numpy.asarray(y_test)



inp = Input(shape=(None, 1))
ls = LSTM(10, activation="sigmoid")(inp)
output = Dense(y.shape[1])(ls)
model = Model(inputs=[inp], outputs=[output])

model.compile(optimizer="adam", loss='mse' )



# early_stp = EarlyStopping(monitor="val_loss", patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,
#                               patience=1, min_lr=0.00000001)

# one_hot_stop = EarlyStopByF1OneHot()

model.fit(x, y, validation_split=0.1, batch_size=10, epochs=1000)

y_predicted = model.predict(x)

for i in range(y_predicted.shape[0]):
    p = np.argmax(y_predicted[i])+1
    y_e = np.argmax(y[i])+1
    print(p, y_predicted[i],
          y_e, y[i], (y_e == p),
          y_unscaled[i])
    # print( "     ", y_predicted[i][0], y[i], y_unscaled[i])

