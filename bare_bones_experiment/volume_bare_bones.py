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

length_of_series = 20


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
from itertools import permutations
# Get all permutations of [1, 2, 3]
perm = permutations([x for x in range(1, )])
# Print the obtained permutations
for i in list(perm):
    print(i)

largest_element = 5
max_count = 3

sequence_len = max_count
num_elements_to_count = 4


from itertools import permutations

ordered_elements = [i/(num_elements_to_count+1) for i in range(1, num_elements_to_count+1)]
perm = permutations(ordered_elements)
perm = list(perm)

x = []
y = []
for p in perm:
    x_temp = []
    y_temp = []
    i = max_count
    for e in p:
        x_temp.append([[e]]*i)
        if i > 1:
            i -= 1
    random.shuffle(x_temp)
    x_temp = [[xtt] for xt in x_temp for xtt in xt]
    dict_element_count = {}
    for e in ordered_elements:
        dict_element_count[e] = 0

    for xt in x_temp:
        dict_element_count[xt[0][0]] = dict_element_count[xt[0][0]] + 1
        y_temp.append([dict_element_count[xt[0][0]]/(max_count+1)])
    print(y_temp)
    x_temp = [xtt for xt in x_temp for xtt in xt]

    x.append(x_temp)
    y.append(y_temp)

one_hot = False
if one_hot:
    y_temp = []
    for i in y:
        y_temp.append(pd.get_dummies(i).values.reshape(-1,))
    y = y_temp
    training_set = list(zip(x, y))*100
    test_set = list(zip(x, y))*10
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
ls  = LSTM(10, activation="sigmoid", return_sequences=True)(inp)
output = TimeDistributed(Dense(1))(ls)
model = Model(inputs=[inp], outputs=[output])

model.compile(optimizer="adam", loss='mse' )



# early_stp = EarlyStopping(monitor="val_loss", patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,
#                               patience=1, min_lr=0.00000001)

# one_hot_stop = EarlyStopByF1OneHot()

model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=100)

y_predicted = model.predict(x_test)

for i in range(y_predicted.shape[0]):
    p = np.argmax(y_predicted[i])+1
    y_e = np.argmax(y_test[i])+1
    print(p, y_predicted[i],
          y_e, y_test[i], (y_e == p),
          y_unscaled[i])
    # print( "     ", y_predicted[i][0], y[i], y_unscaled[i])

