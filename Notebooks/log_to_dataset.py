import ast
file = "/home/known/Desktop/Masters/Code/Actual/memory_capacity_retention_rnns/Notebooks/16_11_2018/12_1_sparsity.log"
from os import listdir
from os.path import isfile, join
directory = "/home/known/Desktop/Masters/Code/Actual/memory_capacity_retention_rnns/Notebooks/16_11_2018"
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

with open('/home/known/Desktop/Masters/Code/Actual/memory_capacity_retention_rnns/Notebooks/pattern_split.txt', 'a') as the_file:
    line_to_write = "folder_root,run_count,f_score,timesteps,sparsity_length,case_type,num_input,num_output,num_patterns_to_recall,num_patterns_total,random_seed,error_when_stopped,num_network_parameters,network_type,training_algorithm,batch_size,epocs,activation_function,num_correctly_identified,architecture,full_network_json ,model_history,full_network ,input_set,output_set"
    the_file.write(line_to_write + '\n')
    for f in onlyfiles:
        if "patterns.log" in f:
            f = open(directory + "/" + f)
            line = f.readline()
            info_seen = False
            single_dataset = [line]
            dataset = []
            count = 0
            while line:
                line = f.readline()
                single_dataset.append(line)
                if "INFO" in line:
                    info_seen = True

                if info_seen:
                    dataset.append(single_dataset)
                    single_dataset = []
                    single_dataset.append(line)
                    info_seen = False
                count += 1
            f.close()

            for element in dataset:
                d = {}
                split_line = element[0].split(";")
                val_acc = element[2].split(";")[0]
                line_to_write = split_line[0] + "," + split_line[1] + "," + split_line[2] + "," + \
                                split_line[3] + "," +split_line[4] + "," + split_line[5] + "," + \
                                split_line[6]+ "," + split_line[7] + "," + split_line[8]  + "," +\
                                split_line[9] + "," + split_line[10]  + "," +  split_line[11] + "," +\
                                split_line[12] + "," + split_line[13]+ "," + split_line[14]+ "," +split_line[15] + "," + \
                                str( val_acc.count('a') +1)+ "," +split_line[17] + "," +split_line[18]
                the_file.write(line_to_write + '\n')
