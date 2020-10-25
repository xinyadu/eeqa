#!/bin/sh

export ACE_DIR=./proc/data/ace-event/processed-data/json
export ACE_PRE_DIR=./trigger_qa_output/epoch0-step0

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args

echo "**************************"
echo "        template 3: des   "
echo "**************************"

echo "=========================================================================================="
echo "                                           real des_query + trigger verb                  "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --train_file $ACE_DIR/toy.json \
  --dev_file $ACE_DIR/toy.json \
  --test_file $ACE_PRE_DIR/trigger_predictions.json \
  --gold_file $ACE_DIR/toy.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_thresh_output \
  --model_dir args_qa_thresh_output/epoch0-step0 \
  --nth_query 5 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
 # To get final results on test, you need to (1) add eval_test; (2) set --model_dir to your model path for test (e.g., epoch-x-step-x)