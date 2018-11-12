import logging

import pandas as pd
from os import listdir
from os.path import isfile, join

from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Base = declarative_base()
# #
# # global engine
# # engine = create_engine('postgresql://masters_user:password@localhost:5432/masters_experiments')
# # import json


# -- Drop table
#
# -- DROP TABLE public.experiments
#
# CREATE TABLE public.experiments (
# 	case_type int ,
# 	num_input int ,
# 	num_output int,
# 	num_patterns_to_recall int,
# 	num_patterns_total int,
# 	timesteps int,
# 	sparsity_length int,
# 	random_seed int ,
# 	run int ,
# 	error_when_stopped float,
# 	num_correctly_identified int ,
# 	input_set text,
# 	output_set text,
# 	architecture text,
# 	num_network_parameters int,
# 	network_type text,
# 	training_algorithm text,
# 	batch_size int,
# 	activation_function text,
# 	full_network text
# )
# WITH (
# 	OIDS=FALSE
# ) ;



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
                      input_set=[0,0,0,0,0],
                      output_set=[0,0,0,0,0],
                      architecture=[0,0,0,0,0],
                      num_network_parameters=0,
                      network_type="lstm",
                      training_algorithm="adam",
                      batch_size=10,
                      activation_function="tanh",
                      full_network="[0,0,0,0,0]",
                      folder_root=""):
    global engine
    df = pd.DataFrame()
    df["case_type"] = [case_type]
    df["num_input"] = [num_input]
    df["num_output"] = [num_output]
    df["num_patterns_to_recall"] = [num_patterns_to_recall]
    df["num_patterns_total"] = [num_patterns_total]
    df["timesteps"] = [timesteps]
    df["sparsity_length"] = [sparsity_length]
    df["random_seed"] = [random_seed]
    df["run"] = [run_count]
    df["error_when_stopped"] = [error_when_stopped]
    df["num_correctly_identified"] = [num_correctly_identified]
    df["input_set"] = [str(input_set)]
    df["output_set"] = [str(output_set)]
    df["architecture"] = [str(architecture)]
    df["num_network_parameters"] = [num_network_parameters]
    df["network_type"] = [network_type]
    df["training_algorithm"] = [training_algorithm]
    df["batch_size"] = [batch_size]
    df["activation_function"] = [activation_function]
    df["full_network"] = [str(full_network)]
    # df.to_sql('experiments', engine, index=False, if_exists='append')
    location = folder_root + "/" + str(run_count) + "_" + str(case_type) + "_" + str(num_input) \
               + "_" + str(num_output) + "_" + str(timesteps) + "_" + str(num_patterns_to_recall) \
               + "_" + str(sparsity_length) \
               + "_" + str(num_patterns_total) + "_" + str(network_type) \
               + "_" + str(activation_function) + "_" + str(num_patterns_total) + ".csv"
    # df.to_csv(location)
    string_to_write = str(folder_root) + "," \
                        + str(run_count) + "," \
                        + str(timesteps) + "," \
                        + str(sparsity_length) + "," \
                        + str(case_type) + "," \
                        + str(num_input) + "," \
                        + str(num_output) + "," \
                        + str(num_patterns_to_recall) + "," \
                        + str(num_patterns_total) + "," \
                        + str(random_seed) + "," \
                        + str(error_when_stopped) + "," \
                        + str(str(architecture)) + "," \
                        + str(num_network_parameters) + "," \
                        + str(network_type) + "," \
                        + str(training_algorithm) + "," \
                        + str(batch_size) + "," \
                        + str(activation_function) + "," \
                        + str(num_correctly_identified) + "," \
                        + str(str(input_set)) + "," \
                        + str(str(output_set)) + "," \
                        + str(full_network) + "\n"
    logging.info(string_to_write)


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


def experiment_exists(case_type=1,
                      num_input=0,
                      num_output=0,
                      num_patterns_to_recall=0,
                      num_patterns_total=0,
                      timesteps=0,
                      sparsity_length=0,
                      run_count=0,
                      network_type="lstm",
                      activation_function="tanh",
                      folder_root=""):
    global logfile
    #
    # onlyfiles = [f for f in listdir(folder_root) if isfile(join(folder_root, f))]
    # location = folder_root + "/" + str(run_count) + "_" + str(case_type) + "_" + str(num_input) \
    #            + "_" + str(num_output) + "_" + str(timesteps) + "_" + str(num_patterns_to_recall) \
    #            + "_" + str(sparsity_length) \
    #            + "_" + str(num_patterns_total) + "_" + str(network_type) \
    #            + "_" + str(activation_function) + "_" + str(num_patterns_total) + ".csv"

    # with open(fname) as f:
    #     content = f.readlines()
    # # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip() for x in content]
    #
    # if location in onlyfiles:
    #     return True
    return False

def say_done():
    logging.info("================ DONE ================")
# def get_dataset(timesteps=1, sparsity=0, num_input=2, num_patterns=2, network_type="lstm", activation_function="tanh",
#                 run=1, folder=""):
#     # global engine
#     # query_str = "select * from experiments where timesteps=" + str(timesteps) \
#     #                        + " and sparsity_length=" + str(sparsity) \
#     #                        + " and num_input=" + str(num_input) \
#     #                        + " and num_patterns_total=" + str(num_patterns) \
#     #                        + " and network_type='" + network_type + "'" \
#     #                        + " and activation_function='" + str(activation_function) + "'" \
#     #                         + " and run=" + str(run)
#     # df = pd.read_sql_query(query_str,
#     #                        con=engine)
#
#     onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
#
#     return df
