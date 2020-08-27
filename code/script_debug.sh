#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json
export ACE_PRE_DIR_ACTION=./trigger_qa_action_output
export ACE_PRE_DIR_VERB=./trigger_qa_verb_output

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries_new.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args

export SQUAD_DIR=../squad_data


# python code/run_args_qa_analysis.py \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/ebug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --gold_file $ACE_DIR/debug.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir_1 ace_args_qa_debug_output \
#   --output_dir_2 ace_args_qa_debug_output \
#   --nth_query_1 3 \
#   --nth_query_2 5 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \

python code/run_args_qa.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/debug.json \
  --dev_file $ACE_DIR/debug.json \
  --test_file $ACE_DIR/debug.json \
  --gold_file $ACE_DIR/debug.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 1 \
  --output_dir ace_args_qa_debug_output \
  --nth_query 3 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

# python code/run_args_qa_unseen.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/debug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --gold_file $ACE_DIR/debug.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_unseen_args_qa_debug_output \
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

# for i in {0..3}
# do

# printf "\n\n\n\n"
# echo "                          ##########"
# echo "                          $i-th try "
# echo "                          ##########"

# echo "=========================================================================================="
# echo "                                           ensemble eval (test time)                                  "
# echo "=========================================================================================="

# python code/run_args_qa_ensem.py \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/debug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --gold_file $ACE_DIR/debug.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir_1 ace_args_qa_debug_output_$i \
#   --output_dir_2 ace_args_qa_debug_output_$i \
#   --nth_query_1 3 \
#   --nth_query_2 5 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \

# done

# echo "=========================================================================================="
# echo "                                           ace_args_qa                                           "
# echo "=========================================================================================="

# python code/run_args_qa_ensem.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/debug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --gold_file $ACE_DIR/debug.json \
#   --train_batch_size 32 \
#   --eval_batch_size 32  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir ace_args_qa_debug_output \
#   --nth_query 4 \
#   --normal_file $TEMPLATE_DIR/arg_queries.csv \
#   --des_file $TEMPLATE_DIR/description_queries.csv \
#   --eval_test \
#   --eval_per_epoch 10 \
#   --max_seq_length 70 \
#   --n_best_size 5 \
#   --max_answer_length 3 \
#   --larger_than_cls \
#   # --add_if_trigger_embedding \


# echo "=========================================================================================="
# echo "                                           squad                                           "
# echo "=========================================================================================="

# export SQUAD_DIR=../dataset

# python code/run_squad.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/sample_train-v2.0.json \
#   --dev_file $SQUAD_DIR/sample_train-v2.0.json \
#   --test_file $SQUAD_DIR/sample_train-v2.0.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 1e-6 \
#   --num_train_epochs 2 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --eval_metric f1 \
#   --output_dir squad2_debug_output \
#   --version_2_with_negative \

# echo "=========================================================================================="
# echo "                                          trigger_qa                                             "
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/debug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --train_batch_size 32 \
#   --eval_batch_size 32  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir trigger_output \
#   --nth_query 2 \
#   --eval_test \

# echo "=========================================================================================="
# echo "                                          trigger                                            "
# echo "=========================================================================================="

# python code/run_trigger.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/debug.json \
#   --dev_file $ACE_DIR/debug.json \
#   --test_file $ACE_DIR/debug.json \
#   --train_batch_size 32 \
#   --eval_batch_size 32  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir trigger_output \
#   --eval_test \


# export SQUAD_DIR=../dataset/

# echo "=========================================================================================="
# echo "                                          2nd                                             "
# echo "=========================================================================================="


# python code/run_squad_start.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/sample_train-v2.0.json \
#   --dev_file $SQUAD_DIR/sample_train-v2.0.json \
#   --test_file $SQUAD_DIR/sample_train-v2.0.json \
#   --train_batch_size 2 \
#   --eval_batch_size 2  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 4 \
#   --max_seq_length 15 \
#   --doc_stride 15 \
#   --output_dir debug_output \
#   --eval_test \
#   --version_2_with_negative \