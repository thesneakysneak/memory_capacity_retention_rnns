import random

import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score

'''
    Define a random set of unique input to output mappings. No input output pair will correspond to other samples.
    Thus there will never exists a set of input values that will have the same output value.
    E.g. Correct
        0 -> 1
        1 -> 5
        2 -> 3
    E.g. Incorrect
        0 -> 1
        1 -> 1
        2 -> 1
'''

def true_accuracy(y_predict, y_true):
    y_true = [np.round(1/x) for x in y_true]
    y_predict_unscaled = [np.round(1/x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


random.seed(1000)
num_patterns = 10
x = random.sample(range(1, num_patterns+1), num_patterns)
y = random.sample(range(1, num_patterns+1), num_patterns)

x = [1.0/z for z in x]
y = [1.0/z for z in y]

training_set = list(zip(x, y))
training_set = training_set*1000
random.shuffle(training_set)

test_set = list(zip(x, y))
test_set = test_set*100
random.shuffle(test_set)

x_train, y_train = zip(*training_set)
x_test, y_test = zip(*test_set)


x_train = np.array(x_train)
y_train = np.array(y_train)


x_train = x_train.reshape(-1, 1, 1).astype(np.float32)
y_train = y_train.reshape(-1, 1).astype(np.float32)


inp = Input(shape=(1, 1))
ls = LSTM(1)(inp)
output = Dense(1)(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, validation_split=.2, callbacks=[reduce_lr], epochs=100)



