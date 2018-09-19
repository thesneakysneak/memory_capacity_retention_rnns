import pandas as pd
def insert_experiment(engine, case_type, num_input, num_output, num_patterns_to_recall,
                      num_patterns_total, sequence_length, sparsity_length, sparsity_erratic,
                      random_seed, binary_input, run_count, error_when_stopped, num_correctly_identified,
                      input_set, output_set, pattern_input_set, pattern_output_set, num_hidden_layers,
                      num_network_parameters, network_type, training_algorithm, batch_size, activation_function,
                      nodes_per_layer, full_network):
    df = pd.DataFrame()
    df["case_type"] = case_type
    df["num_input"] = num_input
    df["num_output"] = num_output
    df["num_patterns_to_recall"] = num_patterns_to_recall
    df["num_patterns_total"] = num_patterns_total
    df["sequence_length"] = sequence_length
    df["sparsity_length"] = sparsity_length
    df["sparsity_erratic"] = sparsity_erratic
    df["random_seed"] = random_seed
    df["binary_input"] = binary_input
    df["run_count"] = run_count
    df["error_when_stopped"] = error_when_stopped
    df["num_correctly_identified"] = num_correctly_identified
    df["input_set"] = input_set
    df["output_set"] = output_set
    df["pattern_input_set"] = pattern_input_set
    df["pattern_output_set"] = pattern_output_set
    df["num_hidden_layers"] = num_hidden_layers
    df["num_network_parameters"] = num_network_parameters
    df["network_type"] = network_type
    df["training_algorithm"] = training_algorithm
    df["batch_size"] = batch_size
    df["activation_function"] = activation_function
    df["nodes_per_layer"] = nodes_per_layer
    df["full_network"] = full_network
    df.to_sql('experiments', engine, index=False)
