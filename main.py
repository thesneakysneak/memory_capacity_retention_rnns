import generate_dataset as gd
import models as mds

import random
from datetime import datetime

from datetime import datetime

import itertools

import save_result
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
def get_optimal_architecture(input_length,
                             architecture_lower= [],
                             architecture_upper= [],
                             batch_size=1,
                             timesteps=3,
                             network_type="lstm",
                             activation_function="tanh",
                             train_input=[],
                             train_out=[],
                             training_alg="adam"
                             ):
    if not architecture_lower:
        architecture_lower = [input_length]
        architecture_upper = [input_length*2 for i in range(5)]
    model_lower = mds.get_model(architecture=architecture_lower,
                          batch_size=1,
                          timesteps=timesteps,
                          network_type=network_type,
                          activation_function=activation_function)
    print("Training model lower")
    model_lower = mds.train_model(train_input, train_out, model_lower, training_alg=training_alg, batch_size=batch_size)
    trainPredict = model_lower.predict(train_input).ravel()
    f_model_lower = mds.determine_score(trainPredict, train_out)

    model_upper = mds.get_model(architecture=architecture_lower,
                              batch_size=1,
                              timesteps=timesteps,
                              network_type=network_type,
                              activation_function=activation_function)
    print("Training model lower")
    model_upper = mds.train_model(train_input, train_out, model_lower, training_alg=training_alg, batch_size=batch_size)
    trainPredict = model_upper.predict(train_input).ravel()
    f_model_upper = mds.determine_score(trainPredict, train_out)

    if f_model_lower == 0:
        return architecture_lower
    else:
        if architecture_lower[0] == input_length:
            architecture_lower = [input_length*2]
        elif
        return get_optimal_architecture(input_length,
                             architecture_lower,
                             architecture_upper,
                             batch_size,
                             timesteps,
                             network_type,
                             activation_function,
                             train_input,
                             train_out,
                             training_alg
                             )

def single_experiment(engine,
                      case_type,
                      num_input_nodes,
                      num_output_nodes,
                      num_patterns,
                      sequence_length,
                      sparsity_length,
                      architecture,
                      network_type,
                      training_alg,
                      activation_function,
                      batch_size):
    # generate set
    train_input, train_out, input_set, output_set, pattern_input_set, pattern_output_set = \
        gd.get_experiment_set(case_type=1,
                              num_input_nodes=num_input_nodes,
                              num_output_nodes=num_output_nodes,
                              num_patterns=num_patterns,
                              sequence_length=sequence_length,
                              sparsity_length=sparsity_length)
    # train model until converge
    # if model fails to fit, need to search the architecture
    model = mds.get_model(architecture=architecture,
                          batch_size=1,
                          timesteps=sequence_length,
                          network_type=network_type,
                          activation_function=activation_function)

    model = mds.train_model(train_input, train_out, model, training_alg=training_alg, batch_size=batch_size)
    return model.history.history["acc"][0]

def run_experiments():
    """
    This function serves as the orchestration of experiments. As in all experiments, one wishes to keep all parameters
    constant except the variable being investigated. All results will be written to a database.
    :return:
    """
    # First get a neural network that is able to fit the input
    case_type = 1
    num_input_nodes = 1
    num_output_nodes = 4
    num_patterns = 3
    sequence_length = 2
    sparsity_length = 1
    pattern_input_set, random_patterns, input_set = generate_set(num_input_nodes, sequence_length, num_patterns)
    pattern_output_set, random_output, output_set = generate_set(num_output_nodes, 1, num_patterns)

    train_list, train_out = gd.create_equal_spaced_patterns(input_set, output_set, random_patterns, random_output, sparsity_length)

    train_list.shape, train_out.shape

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(100, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(LSTM(20,return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(train_list, train_out, epochs=1000, batch_size=10, verbose=2)

    # stimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)

    # make predictions
    trainPredict = model.predict(train_list).ravel()
    trainPredict


    # Test the effect of increasing sparsity

    # Test the effect of increasing number of patterns

    # Test the effect of increasing number of time steps

    return

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