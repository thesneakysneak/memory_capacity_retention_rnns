import logging
from itertools import zip_longest

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils


import generate_dataset as gd
import recurrent_models

import gc
import random

from datetime import datetime



import database_functions



import keras


import numpy as np



def search_architecture(num_input,
                        num_out,
                        x_train,
                        y_train,
                        batch_size=10,
                        timesteps=3,
                        network_type="lstm",
                        activation_function='tanh',
                        base_architecture=[]
                        ):
    architecture = []
    model = None
    result = {"history" : {"acc" : 0}}
    l1_start = 0
    l2_start = 0
    l3_start = 0
    l4_start = 0
    l5_start = 0
    depth_start = num_input*2
    if len(base_architecture) > 2:
        l1_start = base_architecture[1]
        smallest_architecture = min(base_architecture)
        if smallest_architecture > 1:
            depth_start = smallest_architecture
    if len(base_architecture) > 3:
        l2_start = base_architecture[2]
    if len(base_architecture) > 4:
        l3_start = base_architecture[3]
    if len(base_architecture) > 5:
        l4_start = base_architecture[4]
    if len(base_architecture) > 6:
        l5_start = base_architecture[5]

    for depth in range(depth_start, num_input * 3):
        for l5 in range(l5_start, depth):
            for l4 in range(l4_start, depth):
                for l3 in range(l3_start, depth):
                    for l2 in range(l2_start, depth):
                        upper_bound = depth + 1
                        lower_bound = l1_start
                        if depth < lower_bound:
                            upper_bound = lower_bound+1

                        for l1 in range(lower_bound, upper_bound):
                            # Stop if accuracy == 1

                            architecture = [num_input, l1, l2, l3, l4, l5, num_out]
                            print("architecture", architecture)
                            architecture = list(filter(lambda a: a != 0, architecture))  # remove 0s

                            model = recurrent_models.get_model(architecture=architecture,
                                                               batch_size=10,
                                                               timesteps=timesteps,
                                                               network_type=network_type,
                                                               activation_function=activation_function)
                            model, result = recurrent_models.train_model(x_train, y_train, model, "adam",
                                                                         10)
                            validation_acc = np.amax(result.history['acc'])
                            y_predicted = model.predict(x_train, batch_size=batch_size)
                            f_score = recurrent_models.determine_score(y_train, y_predicted, f_only=True)
                            print("f_score", f_score)
                            if f_score >= 1.0:
                                print('Best validation acc of epoch:', validation_acc, "architecture", architecture)
                                return model, result, architecture, f_score


    return model, result, architecture





def investigate_number_of_patterns():
    activation_functions = ["tanh", "sigmoid", "elu",
                            "relu", "exponential", "softplus",
                            "softsign", "hard_sigmoid", "linear"]
    network_type = ["lstm", "gru", "elman_rnn", "jordan_rnn"]
    # Variable we are investigating
    num_patterns = [x for x in range(5, 10)]

    # constants
    sequence_length = 1
    sparsity_length = 0

    import recurrent_models
    from recurrent_models import data

    for i in num_patterns:
        num_input_nodes = i
        print("     num_input_nodes ", i, "output_nodes", (2**num_input_nodes)**sequence_length, "num_patterns", (2**num_input_nodes)**sequence_length)

        # masters_user, password, experiment1_num_patterns
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=(2**num_input_nodes)**sequence_length,
                                  num_patterns=(2**num_input_nodes)**sequence_length,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)

        best_model = search_architecture(num_input_nodes,
                                         2 ** num_input_nodes,
                                         train_input,
                                         train_out,
                                         batch_size=10,
                                         timesteps=1,
                                         network_type="lstm",
                                         activation_function='tanh')
        print(best_model.summary())
        keras.backend.clear_session()



def test_generate_dataset():
    timesteps = [x for x in range(1, 10)]

    print("========================================================================================")
    print("========================================================================================")
    # Keep input size and num patterns constant. Increase timesteps
    for i in timesteps:
        num_input_nodes = 1
        sequence_length = i

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = 2
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    print("========================================================================================")
    print("========================================================================================")
    # Keep timesteps constant. Increase input size
    input_size = [x for x in range(1, 10)]

    for i in input_size:
        num_input_nodes = i
        sequence_length = 2

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = 2
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    # Keep input size and timesteps constant, increase num patterns
    print("========================================================================================")
    print("========================================================================================")


    for i in [x for x in range(2, 10)]:
        num_input_nodes = 10
        sequence_length = 1

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = 0
        num_patterns = i
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)


    # Keep input size,  num patterns and timesteps constant, increase sparsity
    print("========================================================================================")
    print("========================================================================================")

    for i in [x for x in range(2, 10)]:
        num_input_nodes = 10
        sequence_length = 1

        # num_output_nodes = (2**num_input_nodes)**sequence_length
        # num_patterns = (2**num_input_nodes)**sequence_length
        sparsity_length = i
        num_patterns = 3
        num_output_nodes = num_patterns
        # generate set
        train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
            gd.get_experiment_set(case_type=1,
                                  num_input_nodes=num_input_nodes,
                                  num_output_nodes=num_output_nodes,
                                  num_patterns=num_patterns,
                                  sequence_length=sequence_length,
                                  sparsity_length=sparsity_length)

    print("train_input", train_input.shape)
    print("train_input", len(train_input))
    print("train_out", train_out.shape)
    print("train_input", len(train_out))


def run_experiment_sparsity(run, case_type = 1, num_input_nodes = 1, num_output_nodes = 4,
                   timesteps = 1, sparsity_length = 0, num_patterns=0, smallest_architecture=[],
                   activation_function="tanh", network_type="lstm",
                            folder_root="sparsity"):

    """
    Look at the decay in F-score
    :param run:
    :param case_type:
    :param num_input_nodes:
    :param num_output_nodes:
    :param timesteps:
    :param sparsity_length:
    :param num_patterns:
    :param smallest_architecture:
    :param activation_function:
    :param network_type:
    :param folder_root:
    :return:
    """

    print("=======================================================")
    print("=====                                              ====")
    print("=====            Sparsity                          ====")
    print("=====                                              ====")
    print("=====                                              ====")
    print("=======================================================")
    print("run", run, "activation function", activation_function,
          "network", network_type, "sparsity", sparsity_length,
          "num_patterns", num_patterns, "timesteps", timesteps)


    dt = datetime.now()
    random_seed = dt.microsecond
    random.seed(random_seed)


    train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
        gd.get_experiment_set(case_type=1,
                              num_input_nodes=num_input_nodes,
                              num_output_nodes=num_output_nodes,
                              num_patterns=num_patterns,
                              sequence_length=timesteps,
                              sparsity_length=sparsity_length
                              )


    model = recurrent_models.get_model(architecture=smallest_architecture,
                                       batch_size=10,
                                       timesteps=timesteps,
                                       network_type=network_type,
                                       activation_function=activation_function)
    best_model, result = recurrent_models.train_model(train_input, train_out, model, "adam", 10)
    validation_acc = np.amax(result.history['acc'])
    y_predicted = best_model.predict(train_input, batch_size=10)
    f_score = recurrent_models.determine_score(train_out, y_predicted, f_only=True)
    print("f_score", f_score)
    print('Best validation acc of epoch:', validation_acc, "architecture", smallest_architecture)

    print(best_model.summary())
    validation_acc = np.amax(result.history['acc'])

    num_available_patterns = (2 ** num_input_nodes) ** timesteps
    database_functions.insert_experiment(case_type=1,
                                         num_input=num_input_nodes,
                                         num_output=num_output_nodes,
                                         num_patterns_to_recall=num_patterns,
                                         num_patterns_total=num_available_patterns,
                                         timesteps=timesteps,
                                         sparsity_length=sparsity_length,
                                         random_seed=random_seed,
                                         run_count=run,
                                         error_when_stopped=validation_acc,
                                         num_correctly_identified=0,
                                         f_score=str(f_score),
                                         network_type=network_type,
                                         training_algorithm="adam",
                                         batch_size=10,
                                         input_set=str(train_input),
                                         output_set=str(train_out),
                                         architecture=str(best_model.to_json()),
                                         num_network_parameters=best_model.count_params(),
                                         activation_function=activation_function,
                                         full_network_json=str(best_model.to_json()),
                                         full_network=list(best_model.get_weights()),
                                         folder_root=folder_root,
                                         model_history=str(result.history)
                                         )
    keras.backend.clear_session()
    gc.collect()


def run_experiment(run, case_type = 1, num_input_nodes = 1, num_output_nodes = 4,
                   timesteps = 1, sparsity_length = 0, num_patterns=0, smallest_architecture=[], folder_root="timesteps"):

    activation_functions = ["softmax",
                            "elu", "selu", "softplus",
                            "softsign", "tanh", "sigmoid",
                            "hard_sigmoid",
                            "relu",
                              "linear"]
    network_types = ["lstm", "gru", "elman_rnn", ] # "jordan_rnn"
    smallest_architecture_sum = 10000000
    new_smallest = []
    for network_type in network_types:
        for activation_function in activation_functions:
            print("=======================================================")
            print("=====                                              ====")
            print("=====                                              ====")
            print("=====                                              ====")
            print("=====                                              ====")
            print("=======================================================")
            print("run", run, "activation function", activation_function,
                  "network", network_type, "sparsity", sparsity_length,
                  "num_patterns", num_patterns, "timesteps", timesteps)

            # TODO Revisit
            exists = database_functions.experiment_exists(
                                            case_type=1,
                                            num_input=num_input_nodes,
                                            num_output=num_output_nodes,
                                            num_patterns_to_recall=num_patterns,
                                            num_patterns_total=num_patterns,
                                            timesteps=timesteps,
                                            sparsity_length=sparsity_length,
                                            run_count=run,
                                            network_type=network_type,
                                            activation_function=activation_function,
                                            folder_root=folder_root)

            if not exists:
                dt = datetime.now()
                random_seed = dt.microsecond
                random.seed(random_seed)

                train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
                    gd.get_experiment_set(case_type=1,
                                          num_input_nodes=num_input_nodes,
                                          num_output_nodes=num_output_nodes,
                                          num_patterns=num_patterns,
                                          sequence_length=timesteps,
                                          sparsity_length=sparsity_length
                                          )

                best_model, result, architecture, f_score = search_architecture(num_input_nodes,
                                                                       num_output_nodes,
                                                                       train_input,
                                                                       train_out,
                                                                       batch_size=10,
                                                                       timesteps=timesteps,
                                                                       network_type=network_type,
                                                                       activation_function=activation_function,
                                                                       base_architecture=smallest_architecture)
                print(best_model.summary())
                validation_acc = np.amax(result.history['acc'])

                num_available_patterns = (2 ** num_input_nodes) ** timesteps
                count_p = best_model.count_params()
                print(count_p)
                database_functions.insert_experiment(case_type = 1,
                                                    num_input = num_input_nodes,
                                                    num_output = num_output_nodes,
                                                    num_patterns_to_recall = num_patterns,
                                                    num_patterns_total = num_available_patterns,
                                                    timesteps = timesteps,
                                                    sparsity_length=sparsity_length,
                                                    random_seed=random_seed,
                                                    run_count=run,
                                                    error_when_stopped = validation_acc,
                                                    num_correctly_identified = 0,
                                                    num_network_parameters = best_model.count_params(),
                                                    network_type = network_type,
                                                    training_algorithm = "adam",
                                                    f_score = str(f_score),
                                                    activation_function = activation_function,
                                                    architecture = str(architecture),
                                                    input_set=str(train_input),
                                                    output_set=str(train_out),
                                                    batch_size = 10,
                                                    full_network_json = str(best_model.to_json()),
                                                    full_network = list(best_model.get_weights()),
                                                    folder_root=folder_root,
                                                    model_history=str(result.history))

                keras.backend.clear_session()
                architecture_sum = sum(architecture)
                if smallest_architecture_sum >  architecture_sum:
                    smallest_architecture_sum = architecture_sum
                    new_smallest = architecture
    gc.collect()
    return new_smallest


def experiment_loop(run, num_input_nodes_bounds, sparsity_length_bounds, timesteps_bounds, num_patterns_bounds, experiment_type="num_nodes"):
    run = run
    smallest_architecture = []
    # Variable we are investigating
    case_type = 1
    timesteps = 1
    sparsity_length = 0

    # Test effect of increasing num input nodes. All else constant
    print(experiment_type, "num_nodes", experiment_type == "num_nodes")
    if experiment_type == "num_nodes":
        print("Yass")
        for num_input_nodes in num_input_nodes_bounds:
            num_available_patterns = 2 #(2 ** num_input_nodes) ** timesteps
            smallest_architecture = run_experiment(run,
                                                   case_type=1,
                                                   num_input_nodes=num_input_nodes,
                                                   num_output_nodes=2,
                                                   timesteps=1,
                                                   sparsity_length=0,
                                                   num_patterns=2,
                                                   smallest_architecture=smallest_architecture,
                                                   folder_root="num_nodes")

    num_input_nodes = 3
    smallest_architecture = []
    # Test effect of increasing sparsity. All else constant
    if experiment_type =="sparsity":
        ## TODO Revisit
        smallest_architecture_sum = 10000000
        new_smallest = []
        activation_functions = ["softmax",
                                "elu", "selu", "softplus",
                                "softsign", "tanh", "sigmoid",
                                "hard_sigmoid",
                                "relu",
                                "linear"]
        network_types = ["lstm", "gru", "elman_rnn", ]  # "jordan_rnn"

        architecture=[10, 10]
        for network_type in network_types:
            for activation_function in activation_functions:
                # train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
                #     gd.get_experiment_set(case_type=1,
                #                           num_input_nodes=10,
                #                           num_output_nodes=10,
                #                           num_patterns=10,
                #                           sequence_length=1,
                #                           sparsity_length=0
                #                           )
                # best_model, result, architecture, f_score = search_architecture(2, 4,train_input,train_out,
                #                                                        batch_size=10,
                #                                                        timesteps=timesteps,
                #                                                        network_type=network_type,
                #                                                        activation_function=activation_function,
                #                                                        base_architecture=[])

                train_input=  train_out=  input_set=  output_set=  pattern_input_set=  pattern_output_set = []
                for sparsity_length in sparsity_length_bounds:
                    run_experiment_sparsity(run, case_type=case_type,
                                                            num_input_nodes=10,
                                                            num_output_nodes=10,
                                                            timesteps=1,
                                                            sparsity_length=sparsity_length,
                                                            num_patterns=9-1,
                                                            smallest_architecture=architecture,
                                                            folder_root="sparsity",
                                                            activation_function = activation_function,
                                                            network_type = network_type)
    # Test effect of increasing timesteps. All else constant
    num_input_nodes = 3
    sparsity_length = 0
    smallest_architecture = []
    if experiment_type == "timesteps":
        for timesteps in timesteps_bounds:
            smallest_architecture = run_experiment(run,
                           case_type=case_type,
                           num_input_nodes=1,
                           num_output_nodes=(2**1),
                           timesteps=timesteps,
                           sparsity_length=0,
                           num_patterns=(2**1),
                           smallest_architecture=smallest_architecture,
                           folder_root="timesteps")
            # smallest_architecture = run_experiment(run,
            #                                        case_type=case_type,
            #                                        num_input_nodes=1,
            #                                        num_output_nodes=(2 ** 1) ** timesteps,
            #                                        timesteps=timesteps,
            #                                        sparsity_length=0,
            #                                        num_patterns=(2 ** 1) ** timesteps,
            #                                        smallest_architecture=smallest_architecture,
            #                                        folder_root="timesteps")

    # Test effect of increasing num_patterns. All else constant
    smallest_architecture = []

    if experiment_type == "patterns":
        for num_patterns in num_patterns_bounds:
            smallest_architecture = run_experiment(run, case_type=case_type,
                           num_input_nodes=10,
                           num_output_nodes=num_patterns,
                           timesteps=1,
                           sparsity_length=0,
                           num_patterns=num_patterns,
                           smallest_architecture=smallest_architecture,
                           folder_root="patterns")

    return

def test_loop():
    case_type = 1
    num_input_nodes = 1
    num_output_nodes = 4
    num_patterns = 1
    timesteps = 1
    sparsity_length = 0

    activation_functions = ["softmax",
                            "elu", "selu", "softplus",
                            "softsign", "tanh", "sigmoid",
                            "hard_sigmoid",
                            "relu",
                              "linear"]
    network_types = ["lstm", "gru", "elman_rnn", ] # "jordan_rnn"

    # Variable we are investigating
    for run in range(1, 31):
        for sparsity_length in range(0, 51):
            for num_input_nodes in range(1, 31):
                for timesteps in range(1, 31):
                    num_available_patterns = (2**num_input_nodes)**timesteps
                    for num_patterns in range(2, num_available_patterns ):
                        for network_type in network_types:
                            for activation_function in activation_functions:
                                print("=======================================================")
                                print("=====                                              ====")
                                print("=====                                              ====")
                                print("=====                                              ====")
                                print("=====                                              ====")
                                print("=======================================================")
                                print("run", run, "activation function", activation_function,
                                      "network", network_type, "sparsity", sparsity_length,
                                      "num_patterns", num_patterns, "timesteps", timesteps)


                                df = database_functions.get_dataset(timesteps=timesteps,
                                                                    sparsity=sparsity_length,
                                                                    num_input=num_input_nodes,
                                                                    num_patterns=num_patterns,
                                                                    network_type=network_type,
                                                                    activation_function=activation_function,
                                                                    run=run)
                                if df.shape[0] == 0:
                                    dt = datetime.now()
                                    random_seed = dt.microsecond
                                    random.seed(random_seed)
                                    num_output_nodes = num_patterns
                                    if sparsity_length > 0:
                                        num_output_nodes += 1
                                    train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
                                        gd.get_experiment_set(case_type=1,
                                                              num_input_nodes=num_input_nodes,
                                                              num_output_nodes=num_output_nodes ,
                                                              num_patterns=num_patterns,
                                                              sequence_length=timesteps,
                                                              sparsity_length=sparsity_length)

                                    best_model, result, architecture = search_architecture(num_input_nodes,
                                                                                           num_output_nodes,
                                                                     train_input,
                                                                     train_out,
                                                                     batch_size=10,
                                                                     timesteps=timesteps,
                                                                     network_type=network_type,
                                                                     activation_function=activation_function)
                                    print(best_model.summary())
                                    validation_acc = np.amax(result.history['acc'])
                                    count_p = best_model.count_params()
                                    print(count_p)
                                    database_functions.insert_experiment(case_type=1,
                                                                          num_input=num_input_nodes,
                                                                          num_output=num_patterns,
                                                                          num_patterns_to_recall=num_patterns,
                                                                          num_patterns_total=num_available_patterns,
                                                                          timesteps=timesteps,
                                                                          sparsity_length=sparsity_length,
                                                                          random_seed=random_seed,
                                                                          run_count=run,
                                                                          error_when_stopped=validation_acc,
                                                                          num_correctly_identified=0,
                                                                          input_set=str(train_input),
                                                                          output_set=str(train_out),
                                                                          architecture=str(best_model.to_json()),
                                                                          num_network_parameters=best_model.count_params(),
                                                                          network_type=network_type,
                                                                          training_algorithm="adam",
                                                                          batch_size=10,
                                                                          activation_function=activation_function,
                                                                          full_network=str(best_model.get_weights()))
                                    keras.backend.clear_session()

from threading import Thread
import math

def chunks(l, n, max_cores):
    """Yield successive n-sized chunks from l."""
    ar = np.array_split(np.array(l), n)
    ar = [list(filter(None, i)) for i in zip_longest(*ar)]
    count = 0
    while len(ar) > max_cores:
        last = ar.pop(-1)
        ar[count].extend(last)
        ar[count] = sorted(ar[count])
        count += 1
    return ar

def spawn_processes(run_commands=True, run=1, experiment_type="all"):
    import os
    import math
    run_commands = str(run_commands)
    print("run_commands", run_commands, run_commands == "True")
    num_input_nodes_bounds = [x for x in range(1, 15)]
    sparsity_length_bounds = [x for x in range(1, 15)]
    timesteps_bounds = [x for x in range(1, 15)]
    num_patterns_bounds = [x for x in range(2, 100)]

    num_cores_per_experiment = 14

    # experiment_loop(run, num_input_nodes_bounds, sparsity_length_bounds, timesteps_bounds, num_patterns_bounds)
    len_sparsity = int(len(sparsity_length_bounds)/num_cores_per_experiment)
    len_patterns  = int(len(num_patterns_bounds)/num_cores_per_experiment)

    bounds_num_input_nodes = chunks(num_input_nodes_bounds, len_sparsity, num_cores_per_experiment)
    bounds_sparsity_length = chunks(sparsity_length_bounds, len_sparsity, num_cores_per_experiment)
    bounds_time_steps = chunks(timesteps_bounds, len_sparsity, num_cores_per_experiment)
    bounds_num_patterns = chunks(num_patterns_bounds, len_patterns, num_cores_per_experiment)

    for thread in range(num_cores_per_experiment):
        import os

        command_str = 'bash -c "python main.py ' + str(thread) \
                      + ' ' + str(run) + ' num_nodes ' + str(bounds_num_input_nodes[thread]) + '" & '
        if (experiment_type == "all" or experiment_type == "num_nodes"):
            print(command_str)
            if run_commands == "True":
                print("Running")
                os.system(command_str)


        command_str = 'bash -c "python main.py ' + str(thread) \
                      + ' ' + str(run) + ' sparsity ' + str(bounds_sparsity_length[thread]) + '" & '
        if (experiment_type == "all" or experiment_type == "sparsity"):
            print(command_str)
            if run_commands == "True":
                print("Running")
                os.system(command_str)


        command_str = 'bash -c "python main.py ' + str(thread) \
                      + ' ' + str(run) + ' timesteps ' + str(bounds_time_steps[thread]) + '" & '
        if (experiment_type == "all" or experiment_type == "timesteps"):
            print(command_str)
            if run_commands == "True":
                print("Running")
                os.system(command_str)


        command_str = 'bash -c "python main.py ' + str(thread) \
                      + ' ' + str(run) + ' patterns ' + str(bounds_num_patterns[thread]) + '" & '
        if (experiment_type == "all" or experiment_type == "patterns"):
            print(command_str)
            if run_commands == "True":
                print("Running")
                os.system(command_str)


import sys
import ast
def main(args):
    str_array = sys.argv[1:]
    thread = str_array.pop(0)
    run = str_array.pop(0)
    experiment_type = str(str_array.pop(0))
    bounds = ""
    while len(str_array) > 0:
        bounds += str_array.pop(0)
        if bounds.endswith("]]"):
            break
    print("while loop ended")
    bounds = ast.literal_eval(bounds)
    print(run, experiment_type, bounds)

    # TODO Refactor when in cloud
    logfile_location = "/nfs2/danny_masters"
    global logfile
    logfile = logfile_location + "/" +str(thread) + "_" + str(run) + "_" + str(experiment_type) + '.log'
    logging.basicConfig(filename=logfile, level=logging.INFO)

    experiment_loop(run=run,
                    num_input_nodes_bounds=bounds,
                    sparsity_length_bounds=bounds,
                    timesteps_bounds=bounds,
                    num_patterns_bounds=bounds,
                    experiment_type=experiment_type)

if __name__ == "__main__":
    if len(sys.argv[1:]) == 0:
        print("Nothing to do")
        # experiment_loop(run=1,
        #                 num_input_nodes_bounds=[2],
        #                 sparsity_length_bounds=[2],
        #                 timesteps_bounds=[],
        #                 num_patterns_bounds=[],
        #                 experiment_type="sparsity")

        # experiment_loop(run=1,
        #                 num_input_nodes_bounds=[4, 5],
        #                 sparsity_length_bounds=[2],
        #                 timesteps_bounds=[],
        #                 num_patterns_bounds=[],
        #                 experiment_type="num_nodes")
        logfile_location = "/home/known/Desktop/Masters/Code/Actual/memory_capacity_retention_rnns/res"  # "/nfs2/danny_masters"
        global logfile
        logfile = logfile_location + "/timesteps.log"
        logging.basicConfig(filename=logfile, level=logging.INFO)

        experiment_loop(run=1,
                        num_input_nodes_bounds=[4, 5],
                        sparsity_length_bounds=[2],
                        timesteps_bounds=[4],
                        num_patterns_bounds=[],
                        experiment_type="timesteps"
                        )
        # experiment_loop(run=1,
        #                 num_input_nodes_bounds=[4, 5],
        #                 sparsity_length_bounds=[2],
        #                 timesteps_bounds=[],
        #                 num_patterns_bounds=[2, 3],
        #                 experiment_type="patterns")
    else:
        print(sys.argv[1:])
        if sys.argv[1:][0] == "spawn":
            if len(sys.argv[1:]) > 1:
                if len(sys.argv[1:]) > 2:
                    print("Yay")
                    spawn_processes(run_commands=sys.argv[1:][1], run=sys.argv[1:][2], experiment_type=sys.argv[1:][3])
                else:
                    print("Nay")
                    spawn_processes(run_commands=sys.argv[1:][1], run=sys.argv[1:][2])
            else:
                spawn_processes(run_commands=False)

            import time

            while True:
                time.sleep(100)
                print("================Still alive================")
        elif len(sys.argv[1:]) > 3:
            print("Experiment", sys.argv[1:])
            main(sys.argv[1:])
