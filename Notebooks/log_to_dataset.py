import ast
file = "/home/known/Desktop/Masters/Code/Actual/memory_capacity_retention_rnns/Notebooks/16_11_2018/12_1_sparsity.log"

f = open(file)
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
    d["folder_root"] = split_line[0]
    d["run_count"] =  split_line[1]
    d["f_score"] =  split_line[2]
    d["timesteps"] =  split_line[3]
    d["sparsity_length"] =  split_line[4]
    d["case_type"] =  split_line[5]
    d["num_input"] =  split_line[6]
    d["num_output"] =  split_line[7]
    d["num_patterns_to_recall"] =  split_line[8]
    d["num_patterns_total"] =  split_line[9]
    d["random_seed"] =  split_line[10]
    d["error_when_stopped"] =  split_line[11]
    d["num_network_parameters"] =  split_line[12]
    d["network_type"] = split_line[13]
    d["training_algorithm"] = split_line[14]
    d["batch_size"] = split_line[15]
    d["epocs"] = val_acc.count('a') +1
    d["activation_function"] = split_line[17]
    d["num_correctly_identified"] = split_line[18]
    d["architecture"] = split_line[19]
    # d["full_network_json" ] = split_line[20]
    # d["model_history"] = split_line[13]
    # d["full_network" ] = split_line
    # d["input_set"] = split_line
    # d["output_set"] = split_line