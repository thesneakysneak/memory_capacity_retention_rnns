from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

import keras
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN


class ResetState(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.model.reset_states()
        # #         print("reset model state", logs)
        #         acc = logs.get("acc")
        #         if acc == 1.0:

        return


def get_model(architecture=[2, 1, 1, 1],
              batch_size=10, timesteps=3,
              network_type="lstm",
              activation_function='tanh'):
    model = Sequential()
    # Hidden layer 1
    return_sequences = False
    if len(architecture) > 3:
        return_sequences = True
    if network_type == "lstm":
        model.add(LSTM(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                       stateful=True, unroll=True, return_sequences=return_sequences))
    elif network_type == "gru":
        model.add(GRU(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=True, return_sequences=return_sequences))
    elif network_type == "elman_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=True, return_sequences=return_sequences))
    elif network_type == "jordan_rnn":
        model.add(
            SimpleRNN(architecture[1], batch_input_shape=(batch_size, timesteps, architecture[0]),
                      stateful=True, unroll=True, return_sequences=return_sequences))

    # Hidden layer how many ever
    for h in range(2, len(architecture)-1):
        print(h)

        return_sequences = False
        if h < len(architecture)-2:
            return_sequences = True
        if network_type == "lstm":
            model.add(LSTM(architecture[h], return_sequences=return_sequences))

        elif network_type == "gru":
            model.add(GRU(architecture[h], return_sequences=return_sequences))
        elif network_type == "elman_rnn":
            model.add(SimpleRNN(architecture[h], return_sequences=return_sequences))

    model.add(Dense(architecture[-1], activation="softmax"))
    return model


def train_model(input_set, output_set, model, training_alg, batch_size):
    model.compile(loss='categorical_crossentropy', optimizer=training_alg, metrics=['accuracy'])

    callbacks = [
        earlystop,
        reset_state
    ]

    model.fit(input_set, output_set, epochs=10, batch_size=batch_size, verbose=1, shuffle=False, callbacks=callbacks)
    return model


def test():
    network_types = ["lstm", "gru", "elman_rnn", "jordan_rnn"]
    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    architecture = []
    import random
    for i in range(100):
        architecture = []
        num_layers = random.randint(1, 5)
        timesteps = random.randint(1, 300)
        batch_size = random.randint(1, 30)
        nn_index = random.randint(0, len(network_types) - 1)
        activation_index = random.randint(0, len(activation_functions) - 1)
        for i in range(num_layers):
            nodes_in_layer = random.randint(3, 1000)
            architecture.append(nodes_in_layer)
        print("architecture",architecture,
              "batch_size",batch_size,
              "timesteps",timesteps,
              "network_type", network_types[nn_index],
              "activation_function", activation_functions[activation_index])
        model = get_model(architecture=architecture,
                          batch_size=batch_size,
                          timesteps=timesteps,
                          network_type=network_types[nn_index],
                          activation_function=activation_functions[activation_index])
        print(model.summary())


earlystop = EarlyStopping(monitor='loss',  # loss
                          patience=10,
                          verbose=1,
                          min_delta=0.05,
                          mode='auto')
reset_state = ResetState()

if __name__ == "main":
    test()
