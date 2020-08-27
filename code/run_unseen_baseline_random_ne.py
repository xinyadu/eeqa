# baseline random ne
import json
import random

def main():
    random.seed(42)

    unseen_args_file = "./dygiepp/data/ace-event/processed-data/unseen_args"
    unseen_args = {}
    with open(unseen_args_file, "r", encoding='utf-8') as file:
        for line in file:
            unseen_args[line.strip()] = []

    gold_arg = []
    pred_arg = []
    test_file = "./dygiepp/data/ace-event/processed-data/json/test_convert.json"
    with open(test_file, "r", encoding='utf-8') as file:
        example_id = 0
        for line in file:
            example = json.loads(line)
            sentence, events, s_start, entitys = example["sentence"], example["event"], example["s_start"], example["ner"]
            for event in events:
                arguments = event[1:]
                for arg in arguments:
                    arg_type = arg[2]
                    arg_offset = arg[:2]
                    if arg_type in unseen_args:
                        random_ne_offset = entitys[random.randint(0, len(entitys)-1)][:2]
                        pred_arg.append([example_id, arg_type, random_ne_offset])
                        gold_arg.append([example_id, arg_type, arg_offset])
                    
            example_id += 1

    # import ipdb; ipdb.set_trace()
    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    for argument in pred_arg: pred_arg_n += 1
    # gold_arg_n     
    for argument in gold_arg: gold_arg_n += 1
    # pred_in_gold_n
    for argument in pred_arg:
        if argument in gold_arg:
            pred_in_gold_n += 1
    # gold_in_pred_n
    for argument in gold_arg:
        if argument in pred_arg:
            gold_in_pred_n += 1

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_arg_n != 0: prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else: prec_c = 0
    if gold_arg_n != 0: recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else: recall_c = 0
    if prec_c or recall_c: f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else: f1_c = 0

    print("p_c: %.2f, r_c: %.2f, f1_c: %.2f" % (prec_c, recall_c, f1_c))
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
