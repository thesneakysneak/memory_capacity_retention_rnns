from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, nb_epoch=1000)
print(model.predict_proba(X))
"""
[[ 0.0033028 ]
 [ 0.99581173]
 [ 0.99530098]
 [ 0.00564186]]
"""


# Okay and now for our code

X = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
y = np.array([[0],[1],[1],[0]])

inp = Input(shape=(None, 2))
ls = SimpleRNN(10, return_sequences=True)(inp)
ls2 = SimpleRNN(10)(ls)
output = Dense(len(y[0]))(ls2)
model = Model(inputs=[inp], outputs=[output])



# create and fit the LSTM network
# model = Sequential()
# model.add(SimpleRNN(100, input_shape=(None, 1)))
# model.add(SimpleRNN(100, input_shape=(None, 1)))
# model.add(Dense(len(y[0])))

model.compile(loss='mean_squared_error', optimizer='adam')


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.00001)

model.fit(X, y, validation_split=.2, callbacks=[reduce_lr], epochs=1000)
print(model.predict(X))