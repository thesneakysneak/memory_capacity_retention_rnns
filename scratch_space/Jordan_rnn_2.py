import random

import numpy
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import SimpleRNN, Dense

x = [0] * 5000
y = [0] * 5000



for i in range(5000):
    len = 5000
    k = random.randint(5, 45)
    set_of_nums = random.sample([1, 2] * 10000, (len - k)) + [3] * k
    random.shuffle(set_of_nums)
    x[i] = numpy.array(set_of_nums).reshape(-1, 1).astype(np.float32)
    y[i] = numpy.array(1. / k).astype(numpy.float32)


x = numpy.array(x)
y = numpy.array(y)

inp = Input(shape=(5000, 1))
prev_out = Input(shape=(5000, 5))
ls = SimpleRNN(1)(inp)
output = Dense(1)(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, validation_split=.2, callbacks=[reduce_lr])
