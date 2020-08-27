# template 1: what is the [arg]

# template 2: what is the [arg] in [event trigger]

# template 3: description

from os import path
import json
import collections

def print_order_dict(order_dict):
    writer = open(path.join(output_dir, "description_queries.txt"), "w")
    for key in sorted(order_dict.keys()):
        # print(key, order_dict[key])
        for arg in order_dict[key]:
            writer.write("_".join([key, arg]) + ", \n")

# event_types = collections.OrderedDict()
event_types = dict()
input_dir = "./data/ace-event/processed-data/json"
output_dir= "./data/ace-event/processed-data/"
for fold in ["train", "dev", "test"]:
# for fold in ["train", "dev"]:
    # g_convert = open(path.join(output_dir, fold + "_convert.json"), "w")
    with open(path.join(input_dir, fold + ".json"), "r") as g:
        for line in g:
            line = json.loads(line)
            sentences = line["sentences"]
            ner = line["ner"]
            relations = line["relations"]
            events = line["events"]
            sentence_start = line["sentence_start"]
            doc_key = line["doc_key"]

            assert len(sentence_start) == len(ner) == len(relations) == len(events) == len(sentence_start)
            for sent_events in events:
                if sent_events:
                    for sent_event in sent_events:
                        trigger = sent_event[0]
                        arguments = sent_event[1:]
                        if trigger[1] not in event_types:
                            event_types[trigger[1]] = []

                        for arg in arguments:
                            if arg[-1] not in event_types[trigger[1]]:
                                event_types[trigger[1]].append(arg[-1])
                                if fold == "test":
                                    print("!!!", trigger[1], arg[-1])
                                if arg[-1] == "Person" and trigger[1] == "Life.Die":
                                    print("!!!???")
                                    # import ipdb; ipdb.set_trace()
                                if arg[-1] == "Place" and trigger[1] == "Movement.Transport":
                                    print("!!!???")
                                    # import ipdb; ipdb.set_trace()

# import ipdb; ipdb.set_trace()

print_order_dict(event_types)
