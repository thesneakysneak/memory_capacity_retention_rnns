import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
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



x=[0]*5000
y=[0]*5000
y_unscaled=[0]*5000
import random
for i in range(5000):
    len = 50
    k = random.randint(1,10)
    set_of_nums = random.sample([1,2]*100,(len-k)) + [3]*k
    random.shuffle(set_of_nums)
    x[i] = numpy.array(set_of_nums).reshape(-1,1).astype(np.float32)
    y[i] = numpy.array(1./k).astype(np.float32)
    y_unscaled[i] = numpy.array(k).astype(np.float32)
x=numpy.array(x)
y=numpy.array(y)
#
# length_ = 50
#
# x = [0] * length_
# y = [0] * length_
#
# possible = []
# for i in range(50):
#     k = random.randint(1, 5)
#     possible.append(k)
#     set_of_nums = random.sample([1, 2] * length_, (length_ - k)) + [3] * k
#     random.shuffle(set_of_nums)
#     x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
#     y[i] = numpy.array(1. / k).astype(np.float32)
#
# x = numpy.array(x)
# y = numpy.array(y)

inp = Input(shape=(50, 1))
ls = LSTM(1)(inp)
output = Dense(1)(ls)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

early_stp = EarlyStopping(monitor="val_loss", patience=200)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)

model.fit(x, y, validation_split=.2, callbacks=[reduce_lr, early_stp], epochs=1000)

y_predicted = model.predict(x)
for i in range(y_predicted.shape[0]):
    print(convert_to_closest(1 / y_predicted[i][0], set([x for x in range(1, 46)])), convert_to_closest(1 / y[i], set([x for x in range(1, 46)])), y_unscaled[i])



