#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json
export ACE_PRE_DIR_ACTION=./trigger_qa_action_output
export ACE_PRE_DIR_VERB=./trigger_qa_verb_output

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries_new.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args


echo "=========================================================================================="
echo "                                      write the argument predictions  (ensemble eval (test time))                                     "
echo "=========================================================================================="


python code/run_args_qa_analysis.py \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
  --test_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir_1 args_qa_argquery_output_thresh_1 \
  --output_dir_2 args_qa_desquery_output_thresh_1 \
  --nth_query_1 3 \
  --nth_query_2 5 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \


