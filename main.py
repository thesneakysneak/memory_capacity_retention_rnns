
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils


import generate_dataset as gd
import recurrent_models


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
    l1_start = 1
    l2_start = 0
    l3_start = 0
    l4_start = 0
    l5_start = 0
    depth_start = 1
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
                        if depth > int(np.ceil(num_input/2)) and depth>lower_bound:
                            upper_bound = int(np.ceil(num_input/2)) + 1

                        for l1 in range(lower_bound, upper_bound):
                            # Stop if accuracy == 1

                            architecture = [num_input, l1, l2, l3, l4, l5, num_out]
                            print("architecture", architecture)
                            architecture = list(filter(lambda a: a != 0, architecture))  # remove 0s

                            model = recurrent_models.get_model(architecture=architecture,
                                                               batch_size=batch_size,
                                                               timesteps=timesteps,
                                                               network_type=network_type,
                                                               activation_function=activation_function)
                            model, result = recurrent_models.train_model(x_train, y_train, model, "adam",
                                                                         batch_size)
                            validation_acc = np.amax(result.history['acc'])
                            y_predicted = model.predict(x_train, batch_size=batch_size)
                            f_score = recurrent_models.determine_score(y_train, y_predicted, f_only=True)
                            print("f_score", f_score)
                            if f_score >= 1.0:
                                print('Best validation acc of epoch:', validation_acc, "architecture", architecture)
                                return model, result, architecture


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
    from recurrent_models import get_lstm
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


def run_experiment(run, case_type = 1, num_input_nodes = 1, num_output_nodes = 4,
                   timesteps = 1, sparsity_length = 0, num_patterns=0, smallest_architecture=[]):

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
                                          num_output_nodes=num_output_nodes,
                                          num_patterns=num_patterns,
                                          sequence_length=timesteps,
                                          sparsity_length=sparsity_length
                                          )

                best_model, result, architecture = search_architecture(num_input_nodes,
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
                architecture_sum = sum(architecture)
                if smallest_architecture_sum >  architecture_sum:
                    smallest_architecture_sum = architecture_sum
                    new_smallest = architecture
    return new_smallest


def experiment_loop(run):
    smallest_architecture = []
    # Variable we are investigating
    for num_input_nodes in range(2, 31):
        # Test effect of increasing num input nodes. All else constant
        case_type = 1
        timesteps = 1
        sparsity_length = 0

        num_available_patterns = (2 ** num_input_nodes) ** timesteps
        smallest_architecture = run_experiment(run, case_type=case_type, num_input_nodes=num_input_nodes,
                                               num_output_nodes=num_available_patterns,
                                               timesteps=timesteps, sparsity_length=sparsity_length,
                                               num_patterns=num_available_patterns,
                                               smallest_architecture=smallest_architecture)
        for sparsity_length in range(0, 51):
            # Test effect of increasing sparsity. All else constant
            s = [int(x / 2) for x in smallest_architecture]
            run_experiment(run, case_type=case_type, num_input_nodes=num_input_nodes,
                                                   num_output_nodes=num_available_patterns,
                                                   timesteps=timesteps, sparsity_length=sparsity_length,
                                                   num_patterns=num_available_patterns,
                                                   smallest_architecture=s)
        sparsity_length = 0
        for timesteps in range(1, 31):
            # Test effect of increasing timesteps. All else constant
            run_experiment(run, case_type=case_type, num_input_nodes=num_input_nodes,
                                                   num_output_nodes=num_available_patterns,
                                                   timesteps=timesteps, sparsity_length=sparsity_length,
                                                   num_patterns=num_available_patterns,
                                                   smallest_architecture=smallest_architecture)
        timesteps = 1
        for num_patterns in range(2, num_available_patterns):
            # Test effect of increasing num_patterns. All else constant
            run_experiment(run, case_type=case_type, num_input_nodes=num_input_nodes,
                                                   num_output_nodes=num_patterns,
                                                   timesteps=timesteps, sparsity_length=sparsity_length,
                                                   num_patterns=num_patterns,
                                                   smallest_architecture=[])

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
    # gd.example()
    # investigate_number_of_patterns()
    # test_generate_dataset()
    # database_functions.insert_experiment()
    # test_loop()
    experiment_loop(run=1)

if __name__ == "__main__":
    main()
