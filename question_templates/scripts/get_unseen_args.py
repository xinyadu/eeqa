import json
import collections

input_file = "./data/ace-event/processed-data/description_queries_new.csv"
output_file = "./data/ace-event/processed-data/all_args"

out_f = open(output_file, "w")
arg_dict = collections.OrderedDict()
with open(input_file, "r") as f:
    for line in f:
        arg = line.strip().split(',')[0].split('_')[1]
        wh_word = line.strip().split(',')[1].split()[0]
        if arg not in arg_dict:
            arg_dict[arg] = dict()
            arg_dict[arg]['cnt'] = 1
            arg_dict[arg]['wh_word'] = wh_word
        else:
            arg_dict[arg]['cnt'] += 1

# import ipdb; ipdb.set_trace()


for arg in arg_dict:
    if arg_dict[arg]['wh_word'] == "What":
        out_f.write(arg + " " + str(arg_dict[arg]['cnt']) + " " + arg_dict[arg]['wh_word'] + "\n")

for arg in arg_dict:
    if arg_dict[arg]['wh_word'] == "Who":
        out_f.write(arg + " " + str(arg_dict[arg]['cnt']) + " " + arg_dict[arg]['wh_word'] + "\n")

for arg in arg_dict:
    if arg_dict[arg]['wh_word'] == "Where":
        out_f.write(arg + " " + str(arg_dict[arg]['cnt']) + " " + arg_dict[arg]['wh_word'] + "\n")


