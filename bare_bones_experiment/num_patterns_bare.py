import numpy
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score
import generic_functions as gf
import random

length_of_series = 1000


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(numpy.linalg.norm(x - predicted_value)))



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




def generate_sets(num_patterns, one_hot=False):
    correlated = True
    x = y = None

    while correlated:
        x = random.sample(range(1, num_patterns + 1), num_patterns)
        y = random.sample(range(1, num_patterns + 1), num_patterns)
        # print(x, y)
        if num_patterns == 1:
            correlated = False
        else:
            correlated = gf.are_sets_correlated(x, y)

    #
    x = [z / num_patterns for z in x]

    if one_hot:
        y = np.eye(num_patterns)
    else:
        y = [z / num_patterns for z in y]
    #
    training_set = list(zip(x, y))
    training_set = training_set * 100
    random.shuffle(training_set)
    #
    test_set = list(zip(x, y))
    test_set = test_set * 10
    random.shuffle(test_set)
    #
    x_train, y_train = zip(*training_set)
    x_test, y_test = zip(*test_set)

    # Numpy does not know how to deal with tuples
    x_train = list(x_train)
    y_train = list(y_train)

    x_test = list(x_test)
    y_test = list(y_test)

    #
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 1, 1).astype(np.float32)

    if one_hot:
        y_train = y_train.astype(np.float32)
    else:
        y_train = y_train.reshape(-1, 1).astype(np.float32)
    #
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 1, 1).astype(np.float32)


    if one_hot:
        y_test = y_test.astype(np.float32)
    else:
        y_test = y_test.reshape(-1, 1).astype(np.float32)

    #
    return x_train, y_train, x_test, y_test




x_train, y_train, x_test, y_test = generate_sets(5, one_hot=True)
#

inp = Input(shape=(None, 1))
ls = SimpleRNN(1)(inp)
output = Dense(y_train.shape[-1])(ls)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.00001)
model = Model(inputs=[inp], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=10, validation_split=.2, callbacks=[reduce_lr], epochs=1000)

y_predicted = model.predict(x_test)



# print(gf.true_accuracy_one_hot(y_predicted, y_test))

# Even with the traditional orchestration and one hot encoding, it still fails
for i in range(len(y_predicted)):
    print(np.argmax(y_predicted[i]), np.argmax(y_test[i]))




