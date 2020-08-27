from os import path
import json
import collections

output_dir = "./data/ace-event/processed-data/json"
reader = open(path.join(output_dir, "description_queries.csv"), "r")
writer = open(path.join(output_dir, "description_queries_new.csv"), "w")
for line in reader:
    line = line.strip()
    arg, query = line.split(": ")
    writer.write(",".join([arg, query]) + "\n")
