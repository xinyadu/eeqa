#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json
export ACE_PRE_DIR_ACTION=./trigger_qa_action_output
export ACE_PRE_DIR_VERB=./trigger_qa_verb_output

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries_new.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args

export SQUAD_DIR=../squad_data

for i in {0..10}
do
  
printf "\n\n\n\n"
echo "                          ##########"
echo "                          $i-th try "
echo "                          ##########"

echo "=========================================================================================="
echo "                                           real des_query + trigger verb                                     "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_PRE_DIR_VERB/trigger_predictions.json \
  --test_file $ACE_PRE_DIR_VERB/trigger_predictions.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_desquery_output_thresh_$i \
  --nth_query 5 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

echo "=========================================================================================="
echo "                                          real arg_query + trigger verb                                      "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_PRE_DIR_VERB/trigger_predictions.json \
  --test_file $ACE_PRE_DIR_VERB/trigger_predictions.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 3e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_argquery_output_thresh_$i \
  --nth_query 3 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \


# echo "=========================================================================================="
# echo "                                           real des_query + trigger  action                                   "
# echo "=========================================================================================="

# python code/run_args_qa.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --test_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --gold_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_desquery_output \
#   --nth_query 5 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \



echo "=========================================================================================="
echo "                                           ensemble eval (test time)                                  "
echo "=========================================================================================="

python code/run_args_qa_ensem.py \
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
  --output_dir_1 args_qa_argquery_output_thresh_$i \
  --output_dir_2 args_qa_desquery_output_thresh_$i \
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

done
