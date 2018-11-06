import pandas as pd
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

engine = create_engine('postgresql://masters_user:password@localhost:5432/masters_experiments')


def insert_experiment(case_type,
                      num_input,
                      num_output,
                      num_patterns_to_recall,
                      num_patterns_total,
                      timesteps,
                      sparsity_length,
                      random_seed,
                      run_count,
                      error_when_stopped,
                      num_correctly_identified,
                      input_set,
                      output_set,
                      architecture,
                      num_network_parameters,
                      network_type,
                      training_algorithm,
                      batch_size,
                      activation_function,
                      full_network):
    df = pd.DataFrame()
    df["case_type"] = case_type
    df["num_input"] = num_input
    df["num_output"] = num_output
    df["num_patterns_to_recall"] = num_patterns_to_recall
    df["num_patterns_total"] = num_patterns_total
    df["timesteps"] = timesteps
    df["sparsity_length"] = sparsity_length
    df["random_seed"] = random_seed
    df["run"] = run_count
    df["error_when_stopped"] = error_when_stopped
    df["num_correctly_identified"] = num_correctly_identified
    df["input_set"] = input_set
    df["output_set"] = output_set
    df["architecture"] = architecture
    df["num_network_parameters"] = num_network_parameters
    df["network_type"] = network_type
    df["training_algorithm"] = training_algorithm
    df["batch_size"] = batch_size
    df["activation_function"] = activation_function
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


def get_dataset(timesteps=1, sparsity=0, num_input=2, num_patterns=2, network_type="lstm", activation_function="tanh",
                run=1):
    df = pd.read_sql_query("select * from datasets where timesteps=" + str(timesteps)
                           + " and sparsity=" + str(sparsity)
                           + " and num_input=" + str(num_input)
                           + " and num_patterns=" + str(num_patterns)
                           + " and network_type='" + network_type + "'"
                           + " and activation_function='" + str(activation_function) + "'"
                           + " and run=" + str(run),
                           con=engine)
    return df
