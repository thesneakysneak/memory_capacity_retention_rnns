import os
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
    num_pat.run_num_patterns(total_num_parameters=run_num_exp.total_num_parameters, runner=run_num_exp.runner,
                             thread=run_num_exp.thread)


def run_length_exp():
    time.sleep(5)
    length_exp.run_length_experiment(total_num_parameters=run_length_exp.total_num_parameters,
                                     runner=run_length_exp.runner, thread=run_length_exp.thread)


def run_volume_exp():
    time.sleep(5)
    vol_exp.run_volume_experiment(total_num_parameters=run_volume_exp.total_num_parameters,
                                  runner=run_volume_exp.runner, thread=run_volume_exp.thread)


if __name__ == "__main__":
    """
    Runs all other experiments. Pipes them to the background

    Runner is [1, 2, 3, 4, 5]
    :return:
    """

    if len(sys.argv[1:]) != 0:
        runner = int(sys.argv[1:][0])
        print("runner ", runner)

        random.seed(1000)
        total_num_parameters = gf.divisible_by_all(9)
        print(total_num_parameters)

        total_num_parameters = gf.get_runner_experiments(int(runner), total_num_parameters, num_workers=1)
        if runner == 0:
            total_num_parameters = [504, 4536]
        if runner == 1:
            total_num_parameters = [1008, 4032]
        if runner == 2:
            total_num_parameters = [1512 , 3528]
        if runner == 5:
            total_num_parameters = [2016, 3024]
        if runner == 4:
            total_num_parameters = [2520]

        print(str(total_num_parameters).strip(), total_num_parameters[0])
        if total_num_parameters[0] is None or total_num_parameters[0] == "":
            total_num_parameters = total_num_parameters[1:]
        # for i in range(len(total_num_parameters)):
        run_volume_exp.runner = run_length_exp.runner = run_num_exp.runner = runner
        run_volume_exp.thread = run_length_exp.thread = run_num_exp.thread = 1
        run_volume_exp.total_num_parameters = run_length_exp.total_num_parameters = run_num_exp.total_num_parameters = total_num_parameters
        # t = threading.Thread(name='Running number of patterns experiment ' + str(i), target=run_num_exp)
        # t.start()
        #
        # t = threading.Thread(name='Running length of patterns experiment ' + str(i), target=run_length_exp)
        # t.start()
        #
        #
        # t = threading.Thread(name='Running volume of patterns experiment ' + str(i), target=run_volume_exp)
        # t.start()
        command_str = 'bash -c "python3 length_of_sequence.py ' + \
                      str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",").replace(",,", ",")  \
                      + ' ' + str(runner) + ' ' + str(100) + ' ' + str(False) + '" & '
        print("starting ", command_str)

        command_str = 'bash -c "python3 length_of_sequence.py ' + \
                      str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",").replace(",,", ",") \
                      + ' ' + str(runner) + ' ' + str(100) + ' ' + str(True) + '" & '
        print("starting ", command_str)
        # os.system(command_str)
        # time.sleep(5)
        #
        # command_str = 'bash -c "python3 length_of_sequence.py ' + \
        #               str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
        #               + ' ' + str(runner) + ' ' + str(100) + ' ' + str(True) + '" & '
        # print("starting ", command_str)
        # os.system(command_str)
        # time.sleep(5)
        #
        #
        #
        #
        #
        # total_num_parameters_ = total_num_parameters
        # for total_num_parameters in total_num_parameters_:
        #     command_str = 'bash -c "python3 num_patterns.py ' + \
        #                   str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
        #                   + ' ' + str(runner) + ' ' + str(100) + ' ' + str(False) + '" & '
        #     print("tmux new -s num_patterns_"+str(total_num_parameters) + "_False")
        #     print("starting ", command_str)
        #     # os.system(command_str)
        #     # time.sleep(5)
        #     #
        #     command_str = 'bash -c "python3 num_patterns.py ' + \
        #                   str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
        #                   + ' ' + str(runner) + ' ' + str(100) + ' ' + str(True) + '" & '
        #     print("tmux new -s num_patterns_"+str(total_num_parameters) + "_True")
        #     print("starting ", command_str)
        # os.system(command_str)
        # time.sleep(5)
        #
        #
        #
        #
        command_str = 'bash -c "python3 volume_experiment.py ' + \
                      str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",").replace(",,", ",") \
                      + ' ' + str(runner) + ' ' + str(100) + ' ' + str(False) + '" & '
        print("starting ", command_str)
        # os.system(command_str)
        #
        #
        # command_str = 'bash -c "python3 volume_experiment.py ' + \
        #               str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
        #               + ' ' + str(runner) + ' ' + str(100) + ' ' + str(False) + '" & '
        # print("starting ", command_str)
        # os.system(command_str)
        # time.sleep(5)

        # while True:
        #     time.sleep(10)
        #     print("=============== STILL HERE ======================")

        total_num_parameters = gf.divisible_by_all(9)
        total_num_parameters_ = total_num_parameters
        for total_num_parameters in total_num_parameters_:
            command_str = 'bash -c "python3 num_patterns.py ' + \
                          str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
                          + ' ' + str(runner) + ' ' + str(100) + ' ' + str(False) + '" & '
            print("tmux new -s num_patterns_"+str(total_num_parameters) + "_False")
            print("starting ", command_str)
            # os.system(command_str)
            # time.sleep(5)
            #
            command_str = 'bash -c "python3 num_patterns.py ' + \
                          str(total_num_parameters).replace("[ ", "").replace("[", "").replace("]", "").replace(" ", ",") \
                          + ' ' + str(runner) + ' ' + str(100) + ' ' + str(True) + '" & '
            print("tmux new -s num_patterns_"+str(total_num_parameters) + "_True")
            print("starting ", command_str)

