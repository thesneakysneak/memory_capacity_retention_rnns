import random
import sys
import generic_functions as gf
import num_patterns as num_pat
import length_of_sequence as length_exp
import volume_experiment as vol_exp
import threading
import time

def run_num_exp():
    time.sleep(5)
    num_pat.run_num_patterns(total_num_parameters=run_num_exp.total_num_parameters, runner=run_num_exp.runner, thread=run_num_exp.thread)

def run_length_exp():
    time.sleep(5)
    length_exp.run_length_experiment(total_num_parameters=run_length_exp.total_num_parameters, runner=run_length_exp.runner, thread=run_length_exp.thread)

def run_volume_exp():
    time.sleep(5)
    vol_exp.run_volume_experiment(total_num_parameters=run_volume_exp.total_num_parameters, runner=run_volume_exp.runner, thread=run_volume_exp.thread)


def main():
    """
    Runs all other experiments. Pipes them to the background

    Runner is [1, 2, 3, 4, 5]
    :return:
    """

    if len(sys.argv[1:]) == 0:
        return 0

    runner = sys.argv[1:][0]

    random.seed(1000)
    total_num_parameters = gf.divisible_by_all(30)
    total_num_parameters = gf.get_runner_experiments(runner, total_num_parameters)

    for i in range(len(total_num_parameters)):
        run_volume_exp.total_num_parameters = run_length_exp.runner = run_num_exp.runner = runner
        run_volume_exp.total_num_parameters = run_length_exp.thread = run_num_exp.thread = i
        run_volume_exp.total_num_parameters = run_length_exp.total_num_parameters  = run_num_exp.total_num_parameters = [total_num_parameters[i]]
        t = threading.Thread(name='Running number of patterns experiment ' + str(i), target=run_num_exp)
        t.start()

        t = threading.Thread(name='Running length of patterns experiment ' + str(i), target=run_length_exp)
        t.start()


        t = threading.Thread(name='Running volume of patterns experiment ' + str(i), target=run_volume_exp)
        t.start()