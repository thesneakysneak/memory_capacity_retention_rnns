import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score

x = [0] * 5000
y = [0] * 5000
import random

def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


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







#
# Dynamic counting
#

import numpy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np

x = [0] * 10
y = [0] * 10
import random



def train_generator():
    while True:
        sequence_len = random.randint(50, 70)
        k = random.randint(20, 45)
        set_of_nums = random.sample([1, 2] * 10000, (sequence_len - k)) + [3] * k
        random.shuffle(set_of_nums)
        x = numpy.array(set_of_nums).reshape(1, -1, 1).astype(np.float32)
        y = []
        c = 0.0
        for l in x:
            if l[0] ==3:
                c += 1.0
                y.append(1/c)
            else:
                y.append(0.0)
        # y[i] = numpy.array(1. / k).astype(np.float32)
        yield x, numpy.array(y).reshape(1, -1, 1).astype(np.float32)


model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(1, None, 1), stateful=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(train_generator(), steps_per_epoch=1000, epochs=1000, verbose=1, callbacks=[reduce_lr])
