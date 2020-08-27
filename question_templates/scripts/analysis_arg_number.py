# template 1: what is the [arg]

# template 2: what is the [arg] in [event trigger]

# template 3: description

from os import path
import json
import collections

# event_types = collections.OrderedDict()
# event_types = dict()
ace_dir = "./data/ace-event/processed-data/json"
event_type_argument_type_file = "./data/ace-event/processed-data/arg_queries.csv"

all_count = collections.OrderedDict()
with open(event_type_argument_type_file, "r") as f:
    for line in f:
        event_type_argument_type = line.strip().split(",")[0]
        all_count[event_type_argument_type] = []

# import ipdb; ipdb.set_trace()
# for fold in ["train", "dev", "test"]:
fold = "test"
argument_lengths = []
with open(path.join(ace_dir, fold + "_convert.json"), "r") as g:
    for line in g:
        line = json.loads(line)
        sentence = line["sentence"]
        events = line["event"]
        s_start = line["s_start"]
        
        for event in events:
            event_type = event[0][1]
            count = collections.defaultdict(int)
            for argument in event[1:]:
                argument_type = argument[-1]
                count[argument_type] += 1
                argument_lengths.append(argument[1] - argument[0] + 1)
                if argument_lengths[-1] > 3: print(argument, sentence[argument[0] - s_start:argument[1] + 1 - s_start])

            for argument_type, num in count.items():
                event_type_argument_type = "_".join([event_type, argument_type])
                # if num not in all_count[event_type_argument_type]:
                if event_type_argument_type not in all_count:
                    print(event_type_argument_type, sentence, events)
                    assert ValueError
                else:
                    all_count[event_type_argument_type].append(num)

        # if events:
            # import ipdb; ipdb.set_trace()
for key, item in all_count.items():
    item = collections.Counter(sorted(item))
    if len(item)>1:
        print(key, collections.Counter(item))

argument_lengths_cnt = collections.Counter(sorted(argument_lengths))

import ipdb; ipdb.set_trace()

