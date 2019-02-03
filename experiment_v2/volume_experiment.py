
# https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/

import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score

import recurrent_models

x = [0] * 5000
y = [0] * 5000
import random

def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


for i in range(5000):
    length_ = 5000
    k = random.randint(5, 45)
    k2 = random.randint(5, 45)
    set_of_nums = random.sample([1, 2] * 10000, (length_ - k - k2)) + [3] * k + [4]*k2
    random.shuffle(set_of_nums)
    x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
    y[i] = numpy.array([1. / k, 1. / k2]).astype(np.float32)


x = numpy.array(x)
y = numpy.array(y)

inp = Input(shape=(5000, 1))
ls = SimpleRNN(2)(inp)
output = Dense(2)(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, validation_split=.2, callbacks=[reduce_lr, recurrent_models.earlystop2],   epochs=1000)

y_predict = 1/model.predict(x)
y_predict = [[round(p[0]), round(p[1])] for p in y_predict]



