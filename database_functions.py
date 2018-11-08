import pandas as pd
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

engine = create_engine('postgresql://masters_user:password@localhost:5432/masters_experiments')


def insert_experiment(case_type=1,
                      num_input=0,
                      num_output=0,
                      num_patterns_to_recall=0,
                      num_patterns_total=0,
                      timesteps=0,
                      sparsity_length=0,
                      random_seed=0,
                      run_count=0,
                      error_when_stopped=0.0,
                      num_correctly_identified=0,
                      input_set=0,
                      output_set=0,
                      architecture=[0,0,0,0,0],
                      num_network_parameters=0,
                      network_type="lstm",
                      training_algorithm="adam",
                      batch_size=10,
                      activation_function="tanh",
                      full_network=""):
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
    df["full_network"] = str(full_network)
    df.to_sql('experiments', engine, index=False, if_exists='append')


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
    df = pd.read_sql_query("select * from experiments where timesteps=" + str(timesteps)
                           + " and sparsity_length=" + str(sparsity)
                           + " and num_input=" + str(num_input)
                           + " and num_patterns_total=" + str(num_patterns)
                           + " and network_type='" + network_type + "'"
                           + " and activation_function='" + str(activation_function) + "'"
                           + " and run=" + str(run),
                           con=engine)
    return df
