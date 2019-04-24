import random
import numpy as np
from sklearn.metrics import r2_score
import ExampleRNNs.experiment_constants as const


def true_accuracy(y_predict, y_true):
    y_predict_unscaled = [round(x) for x in y_predict]
    return r2_score(y_predict_unscaled, y_true)


def convert_to_closest(predicted_value, possible_values):
    return min(possible_values, key=lambda x: abs(x - predicted_value))


def get_nodes_in_layer(num_parameters, nn_type):
    if nn_type == const.LSTM:
        return int(num_parameters / 12)
    if nn_type == const.GRU:
        return int(num_parameters / 9)
    return int(num_parameters / 3)


def get_runner_experiments(runner, total_num_parameters):
    total_num_parameters = np.array(total_num_parameters).reshape(-1, 5)
    for i in range(5):
        if i % 2 == 0:
            total_num_parameters[i] = sorted(total_num_parameters[i], reverse=True)
    total_num_parameters = np.transpose(total_num_parameters)
    random.shuffle(total_num_parameters)
    return total_num_parameters[runner]
