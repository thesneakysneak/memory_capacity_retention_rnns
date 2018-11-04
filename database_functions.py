import pandas as pd
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()

engine = create_engine('postgresql://masters_user:password@localhost:5432/masters_experiments')

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

def insert_dataset(timesteps=1, sparsity=0, num_input=2, num_patterns=2,
                   train_input=[[[0, 1]], [[1, 0]]],
                   train_target=[[[0, 1]], [[1, 0]]],
                   input_set=[[[0, 1]], [[1, 0]]],
                   output_set=[[[0, 1]], [[1, 0]]],
                   pattern_input_set=[[[0, 1]], [[1, 0]]],
                   pattern_output_set=[[[0, 1]], [[1, 0]]]):
    df = pd.DataFrame()
    df["timesteps"] = [timesteps]
    df["sparsity"] = [sparsity]
    df["num_input"] = [num_input]
    df["num_patterns"] = [num_patterns]
    df["train_input"] = [str(train_input)]
    df["train_target"] = [str(train_target)]
    df["input_set"] = [str(input_set)]
    df["output_set"] = [str(output_set)]
    df["pattern_input_set"] = [str(pattern_input_set)]
    df["pattern_output_set"] = [str(pattern_output_set)]
    df.to_sql('datasets', engine, index=False, if_exists='append')

def get_dataset(timesteps=1, sparsity=0, num_input=2, num_patterns=2):
    df = pd.read_sql_query('select * from datasets where timesteps={} and sparsity={} and num_input={} and num_patterns={}'\
                           .format(timesteps, sparsity, num_input,num_patterns), con=engine)
    return df


