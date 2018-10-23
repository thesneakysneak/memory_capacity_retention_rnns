import scipy
import random
import numpy as np
from  sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix

# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import pandas as pd
from keras import Sequential
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
import random
import random
from datetime import datetime

import itertools

import save_result

from numpy import array
from numpy import argmax
from keras.utils import to_categorical

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

def all_unique_patterns(input_set, length):
    length = len(input_set)
    array = []
    for k in range(length):
        for i in input_set:
            for j in input_set:
                array.append(i)


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
    possible_patterns = generate_bit_patterns(input_length)

    all_sequences = list(itertools.product(possible_patterns, repeat=sequence_length))
    index_of_set = random.sample(range(0, len(all_sequences)), num_patterns)
    sequences_to_identify = [all_sequences[i] for i in index_of_set]
    random_patterns = [x for x in all_sequences if x not in sequences_to_identify]
    return np.array(all_sequences), np.array(random_patterns), np.array(sequences_to_identify)


def create_equal_spaced_patterns(patterns_to_identify, corresponding_output, random_patterns, random_output,
                                 sparsity_spacing=5, total_input_length=100):
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
    num_r_output = len(random_output)
    num_available_patterns = len(patterns_to_identify)
    while counter <= total_input_length:
        if pattern_count >= num_available_patterns:
            pattern_count = 0
        if counter % (sparsity_spacing + 1) == 0:
            train_list.append(patterns_to_identify[pattern_count])
            train_out.append(corresponding_output[pattern_count])
            pattern_count += 1
        else:
            rand_index_in = random.randint(0, len(random_patterns) - 1)
            print("random_patterns", rand_index_in)
            train_list.append(random_patterns[rand_index_in])
            rand_index_out = random.randint(0, num_r_output - 1)
            print("random_output", rand_index_out)
            train_out.append(random_output[rand_index_out])
        counter += 1

    train_list = np.array(train_list)

    train_out = np.array(train_out)
    train_out = train_out.reshape(train_out.shape[0], train_out.shape[2])
    return train_list,train_out

def generate_one_hot_output(num_patterns):
    # define example
    data = [x for x in range(num_patterns)]
    data = array(data)
    # one hot encode
    encoded = to_categorical(data)
    return encoded

def get_experiment_set(case_type=1, num_input_nodes=3, num_output_nodes=3, num_patterns=3, sequence_length=2,
                       sparsity_length=1):
    pattern_input_set, random_patterns, input_set = generate_set(num_input_nodes, sequence_length, num_patterns)
    # pattern_output_set, random_output, output_set = generate_set(num_output_nodes, sequence_length, num_patterns)
    output = generate_one_hot_output(num_patterns+1)
    output = random.sample(list(output), len(output))
    pattern_output_set = output
    random_patterns = [output.pop(0)]
    output_set = output
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



def determine_score(predicted, test):
    p_categories = [np.argmax(x) for x in predicted]
    t_categories = [np.argmax(x) for x in test]

    conf_mat = confusion_matrix(t_categories, p_categories)
    precision, recall, fbeta_score, beta = precision_recall_fscore_support(t_categories, p_categories, average="micro")

    return precision, recall, fbeta_score, conf_mat

def example():
    case_type = 1
    num_input_nodes = 1
    num_output_nodes = 4
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    pattern_input_set, random_patterns, input_set = generate_set(num_input_nodes, sequence_length, num_patterns)
    pattern_output_set, random_output, output_set = generate_set(num_output_nodes, 1, num_patterns)

    train_list, train_out = create_equal_spaced_patterns(input_set, output_set, random_patterns, random_output, sparsity_length)

    train_list.shape, train_out.shape

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(sequence_length, 1)))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.fit(train_list, train_out, epochs=100, batch_size=1, verbose=2)

    stimator = KerasClassifier(build_fn=model, epochs=model, batch_size=5, verbose=0)

    # make predictions
    trainPredict = model.predict(train_list).ravel()
    trainPredict



def tests():
    # Generate first experiment
    expected = np.array([
                         [[0]], [[0]],
                         [[0]], [[1]],
                         [[1]], [[0]],
                         [[1]], [[1]],
    ])
