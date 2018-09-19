import scipy
import random

import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

import random
from datetime import datetime

import itertools

import save_result

def __print_lists__(train_list, train_out, corresponding_output):
    """
    Used for debugging
    :param train_list:
    :param train_out:
    :param corresponding_output:
    :return:
    """
    count_p = 0
    for i in range(len(train_list)):
        if train_out[i] in corresponding_output:
            print("p ", train_list[i], " --->>> ", train_out[i])
            count_p += 1
        else:
            print("r ", train_list[i], " --->>> ", train_out[i])
    print("Count ", count_p)


def generate_bit_patterns(bit_length=3):
    """
        Generates all possible binary sequences up to bit_length
    :param bit_length:
    :return: array of binary sequences e.g. [[0,0],[0,1],[1,0],[1,1]]
    """
    unique_input_patterns = []

    for bits in itertools.product([0, 1], repeat=bit_length):
        single_input = [bit for bit in bits]
        unique_input_patterns.append(single_input)
    return unique_input_patterns


def generate_set(input_length=3, sequence_length=3, num_patterns=3):
    """
    Generates a set containing all combinations of binary patterns that have input_length.
    Each element is of sequence_length.
    The patterns that should be identified are sampled randomly and removed from random_patterns.
    Example: with input_length=2, sequence_length =2, num_patterns=2
            binary set to select from -> [[0,0],[0,1],[1,0],[1,1]]
            sequence length enhanced set -> [([0, 0], [0, 1]),
                                             ([0, 0], [1, 0]),
                                             ([0, 0], [1, 1]),
                                             ([0, 1], [1, 0]),
                                             ([0, 1], [1, 1]),
                                             ([1, 0], [1, 1])]
            set to identify -> [([0, 1], [1, 1]), ([0, 0], [1, 1])]

    :param input_length: Binary pattern length
    :param sequence_length: Length of combined binary inputs
    :param num_patterns: Num patterns that should be identified correctly

    :return: patterns_to_identify, random_patterns, all_available_patterns
    """
    possible_inputs = generate_bit_patterns(input_length)
    all_available_patterns = list(itertools.combinations(possible_inputs, sequence_length))
    index_of_set = random.sample(range(0, len(all_available_patterns)), num_patterns)
    patterns_to_identify = [all_available_patterns[i] for i in index_of_set]
    random_patterns = [x for x in all_available_patterns if x not in patterns_to_identify]
    return patterns_to_identify, random_patterns, all_available_patterns


def create_equal_spaced_patterns(patterns_to_identify, corresponding_output, random_patterns, random_output,
                                 sparsity_spacing=5):
    """
    Generates a sequence in which the patterns to be identified are separated using the same sparsity.
    E.g.
        patterns_to_identify = [([0, 1], [1, 0]), ([0, 0], [1, 0])]
        corresponding_output = [([0, 1],), ([1, 0],)]
        random_patterns = [([0, 0], [0, 1]), ([0, 0], [1, 1]), ([0, 1], [1, 1]), ([1, 0], [1, 1])]
        random_output = [([0, 0],), ([1, 1],)]
        sparsity_space = 1
    will result in
        r  [0, 0]  --->>>  ([0, 0],)
        r  [1, 1]  --->>>  ([1, 1],)
        r  [0, 1]  --->>>  ([1, 1],)
        p  [1, 0]  --->>>  ([0, 1],)
        r  [0, 1]  --->>>  ([1, 1],)
        r  [1, 1]  --->>>  ([1, 1],)
        r  [0, 0]  --->>>  ([1, 1],)
        p  [1, 0]  --->>>  ([1, 0],)
    :usage:
    train_list, train_out = create_equal_spaced_patterns(
                                 patterns_to_identify,
                                 corresponding_output,
                                 random_patterns,
                                 random_output,
                                 sparsity_spacing=1)

    :param patterns_to_identify:
    :param corresponding_output:
    :param random_patterns:
    :param random_output:
    :param sparsity_spacing:
    :return: train_list, train_out
    """
    train_list = []
    train_out = []
    pattern_count = 0
    counter = 1
    sequence_length = len(patterns_to_identify[0])
    while pattern_count < len(patterns_to_identify):
        if counter % (sparsity_spacing + 1) == 0:
            for x in patterns_to_identify[pattern_count]:
                train_list.append(x)
            for x in range(sequence_length - 1):
                rand_index_out = random.randint(0, len(random_output) - 1)
                train_out.append(random_output[rand_index_out])
            train_out.append(corresponding_output[pattern_count])
            pattern_count += 1
        else:
            rand_index_in = random.randint(0, len(random_patterns) - 1)
            for x in random_patterns[rand_index_in]:
                train_list.append(x)
            for x in range(sequence_length):
                rand_index_out = random.randint(0, len(random_output) - 1)
                train_out.append(random_output[rand_index_out])
        counter += 1
    return train_list, train_out


def get_experiment_set(case_type=1, num_input_nodes=3, num_output_nodes=3, num_patterns=3, sequence_length=2,
                       sparsity_length=1):
    pattern_input_set, random_patterns, input_set = generate_set(num_input_nodes, sequence_length, num_patterns)
    pattern_output_set, random_output, output_set = generate_set(num_output_nodes, 1, num_patterns)

    if case_type == 1:
        train_list, train_out = create_equal_spaced_patterns(pattern_input_set, pattern_output_set, random_patterns,
                                                             random_output, sparsity_length)
    elif case_type == 2:
        train_list, train_out = create_equal_spaced_patterns(pattern_input_set, pattern_output_set, random_patterns,
                                                             output_set, sparsity_length)
    else:
        print("Case ", case_type, "not supported")
        return
        # __print_lists__(train_list, train_out, pattern_output_set)

    return train_list, train_out, input_set, output_set, pattern_input_set, pattern_output_set


def get_model(num_input, num_hidden_layers, num_output, network_type, training_alg, activation_function,
              nodes_per_hidden_layer):
    pass


def train_model(model, train_list, train_out, batch_size):
    model_error_on_converge = 0.0
    return model, model_error_on_converge


def single_experiment(engine,
        case_type, num_input, num_output, num_patterns, sequence_length, sparsity_length,

        num_hidden_layers, network_type, training_alg, activation_function, nodes_per_hidden_layer, batch_size):
    train_list, train_out, input_set, output_set, pattern_input_set, pattern_output_set = get_experiment_set(case_type,
                                                                                                             num_input,
                                                                                                             num_output,
                                                                                                             num_patterns,
                                                                                                             sequence_length,
                                                                                                             sparsity_length)
    model = get_model(num_input, num_hidden_layers, num_output, network_type, training_alg,
                      activation_function, nodes_per_hidden_layer)
    trained_model, model_error_on_converge = train_model(model, train_list, train_out, batch_size)
    num_correct = get_num_patterns_correctly_recalled(trained_model, pattern_input_set, pattern_output_set)
    save_result.insert_experiment(engine, case_type, num_input, num_output, num_patterns,
                      num_patterns_total, sequence_length, sparsity_length, sparsity_erratic=False,
                      random_seed, binary_input, run_count, error_when_stopped, num_correctly_identified,
                      input_set, output_set, pattern_input_set, pattern_output_set, num_hidden_layers,
                      num_network_parameters, network_type, training_algorithm, batch_size, activation_function,
                      nodes_per_layer, full_network)

def get_num_patterns_correctly_recalled(trained_model, pattern_input_set, pattern_output_set):
    num_correct = 0
    for p_index in range(len(pattern_input_set)):
        predicted = None
        for i in pattern_input_set[p_index]:
            print(i)
        num_correct += 1
    #             predicted = trained_model.predict(i)
    #         predicted = [round(x) for x in predicted]
    #         if predicted == pattern_output_set[p_index]:
    #             num_correct += 1
    return num_correct


def main():
    case_type = 1
    num_input_nodes = 3
    num_output_nodes = 3
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    sparsity_erratic = 0
    random_seed = datetime.now().timestamp()
    binary_input = 1

    num_input_nodes
    num_hidden_layers
    num_output_nodes
    network_type
    training_alg
    activation_function
    nodes_per_hidden_layer
    batch_size
    from sqlalchemy import create_engine
    engine = create_engine('postgresql://masters_user:password@localhost:5432/masters_experiments')
    sparsity_erratic = 0
    binary_input = 1
    nodes_per_hidden_layer = [0]
    experiment_count = 0
    nodes_per_hidden_layer = [0]
    max_hidden = 3

    for num_hidden_layers in range(0, max_hidden):
        for num_nodes_in_hidden_layer in range(0, 3):
            nodes_per_hidden_layer[num_hidden_layers] += 1
            for num_output_nodes in range(num_input_nodes):
                for num_patterns in range(1, 2 ^ (num_input_nodes - 1)):
                    for sequence_length in range(1, 3):
                        for sparsity_length in range(1, 3):
                            for network_type in ["elman_rnn", "jordan_rnn", "lstm", "gru"]:
                                for training_alg in ["sgd", "Adam", "RMSPROP", "NAG"]:
                                    for activation_function in ["tanh", "sigmoid", "relu"]:
                                        for case_type in [1, 0]:
                                            batch_size = sequence_length
                                            for run_count in range(0, 10):
                                                random_seed = datetime.now().timestamp()
                                                experiment_count += 1

        if num_hidden_layers != max_hidden - 1:
            nodes_per_hidden_layer.append(0)

    print(experiment_count)
    print(
        "random_seed", random_seed, "\n",
        "run_count", run_count, "\n",
        "case_type", case_type, "\n",
        "num_input_nodes", num_input_nodes, "\n",
        "num_output_nodes", num_output_nodes, "\n",
        "num_patterns", num_patterns, "\n",
        "sequence_length", sequence_length, "\n",
        "sparsity_length", sparsity_length, "\n",
        "sparsity_erratic", sparsity_erratic, "\n",
        "random_seed", random_seed, "\n",
        "binary_input", binary_input, "\n",
        "num_input_nodes", num_input_nodes, "\n",
        "num_hidden_layers", num_hidden_layers, "\n",
        "num_output_nodes", num_output_nodes, "\n",
        "network_type", network_type, "\n",
        "training_alg", training_alg, "\n",
        "activation_function", activation_function, "\n",
        "nodes_per_hidden_layer", nodes_per_hidden_layer, "\n",
        "batch_size", batch_size, "\n")