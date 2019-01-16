import numpy
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *

x = [0] * 5000
y = [0] * 5000
import random

for i in range(5000):
    len = 50
    k = random.randint(5, 45)
    set_of_nums = random.sample([1, 2] * 10000, (len - k)) + [3] * k
    random.shuffle(set_of_nums)
    x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
    y[i] = numpy.array(1. / k).astype(np.float32)


x = numpy.array(x)
y = numpy.array(y)

inp = Input(shape=(50, 1))
ls = LSTM(1)(inp)
output = Dense(1)(ls)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, validation_split=.2)







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
        x[i] = numpy.array(set_of_nums).reshape(1, -1, 1).astype(np.float32)
        y = []
        c = 0.0
        for l in x[i]:
            if l[0] ==3:
                c += 1.0
                y.append(1/c)
            else:
                y.append(0.0)
        # y[i] = numpy.array(1. / k).astype(np.float32)
        yield numpy.array(set_of_nums).reshape(1, -1, 1).astype(np.float32), numpy.array(y).reshape(1, -1, 1).astype(np.float32)


model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(None, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))


model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit_generator(train_generator(), steps_per_epoch=100, epochs=100, verbose=1)
