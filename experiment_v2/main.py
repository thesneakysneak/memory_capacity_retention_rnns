import random
import sys
import experiment_v2.generic_functions as gf

def main():
    """
    Runs all other experiments. Pipes them to the background
    :return:
    """

    if len(sys.argv[1:]) == 0:
        return 0

    runner = sys.argv[1:][0]

    random.seed(1000)
    total_num_parameters = gf.divisible_by_all(30)
    total_num_parameters = gf.get_runner_experiments(runner, total_num_parameters)
    run = runner
