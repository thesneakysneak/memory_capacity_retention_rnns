import generate_dataset as gd
import models as mds

import random
from datetime import datetime

from datetime import datetime

import itertools

import save_result
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

def single_experiment(engine,
                      case_type, num_input_nodes, num_output_nodes, num_patterns, sequence_length, sparsity_length,
                      architecture, network_type, training_alg, activation_function, batch_size):
    train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
        gd.get_experiment_set(case_type=1,
                              num_input_nodes=num_input_nodes,
                              num_output_nodes=num_output_nodes,
                              num_patterns=num_patterns,
                              sequence_length=sequence_length,
                              sparsity_length=sparsity_length)

    model = mds.get_model(architecture=architecture,
                          batch_size=1,
                          timesteps=sequence_length,
                          network_type=network_type,
                          activation_function=activation_function)

    model = mds.train_model(train_input, train_out, model, training_alg=training_alg, batch_size=batch_size)
    return model.history.history["acc"][0]

def main():
    case_type = 1
    num_input_nodes = 3
    num_output_nodes = 2
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    sparsity_erratic = 0
    random_seed = datetime.now().timestamp()
    binary_input = 1

    num_hidden_layers = 1
    network_type = "lstm"
    training_alg = "adam"
    activation_function = "tanh"
    architecture = [num_input_nodes, 2, num_output_nodes]
    batch_size = 10
    gd.example()

if __name__ == "__main__":
    main()