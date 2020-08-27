#!/bin/sh

export ACE_DIR=./proc/data/ace-event/processed-data/json

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args


# echo "=========================================================================================="
# echo "                                          gold arg_query + trigger                        "
# echo "=========================================================================================="

# python code/run_args_qa_unseen.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/toy.json \
#   --dev_file $ACE_DIR/toy.json \
#   --test_file $ACE_DIR/toy.json \
#   --gold_file $ACE_DIR/toy.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_unseen_output \
#   --nth_query 3 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --unseen_arguments_file $UNSEEN_ARG_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \
#   # --add_if_trigger_embedding \


  
echo "=========================================================================================="
echo "                                           gold des_query + trigger                       "
echo "=========================================================================================="

python code/run_args_qa_unseen.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/toy.json \
  --dev_file $ACE_DIR/toy.json \
  --test_file $ACE_DIR/toy.json \
  --gold_file $ACE_DIR/toy.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_unseen_output \
  --nth_query 5 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --unseen_arguments_file $UNSEEN_ARG_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \
  # --add_if_trigger_embedding \


# echo "=========================================================================================="
# echo "                                           gold arg_name + trigger                        "
# echo "=========================================================================================="

# python code/run_args_qa_unseen.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/toy.json \
#   --dev_file $ACE_DIR/toy.json \
#   --test_file $ACE_DIR/toy.json \
#   --gold_file $ACE_DIR/toy.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_unseen_output \
#   --nth_query 1 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --unseen_arguments_file $UNSEEN_ARG_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \