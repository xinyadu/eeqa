#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json
export ACE_PRE_DIR_ACTION=./trigger_qa_action_output
export ACE_PRE_DIR_VERB=./trigger_qa_verb_output

export ARG_QUERY_FILE=./question_templates/arg_queries.csv
export DES_QUERY_FILE=./question_templates/description_queries_new.csv
export UNSEEN_ARG_FILE=./question_templates/unseen_args

export SQUAD_DIR=../squad_data

echo "**************************"
echo "        template 3: des"
echo "**************************"

# echo "=========================================================================================="
# echo "                                           real des_query + trigger action                                       "
# echo "=========================================================================================="

# python code/run_args_qa_thresh.py \
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
#   --output_dir args_qa_output_thresh \
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
echo "                                           real des_query + trigger  verb                                     "
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
  --output_dir args_qa_output_thresh \
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
echo "                                           real des_query  verb                                     "
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
  --output_dir args_qa_output_thresh \
  --nth_query 4 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

  
echo "=========================================================================================="
echo "                                           gold des_query + trigger                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
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
echo "                                           gold des_query                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
  --nth_query 4 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \





echo "**************************"
echo "   template 2: arg_query"
echo "**************************"




# echo "=========================================================================================="
# echo "                                          real arg_query + trigger action                                        "
# echo "=========================================================================================="

# python code/run_args_qa_thresh.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --test_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --gold_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 3e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_output_thresh \
#   --nth_query 3 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \

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
  --output_dir args_qa_output_thresh \
  --nth_query 3 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

echo "=========================================================================================="
echo "                                          real arg_query verb                                      "
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
  --output_dir args_qa_output_thresh \
  --nth_query 2 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \


echo "=========================================================================================="
echo "                                          gold arg_query + trigger                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
  --nth_query 3 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

echo "=========================================================================================="
echo "                                          gold arg_query                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
  --nth_query 2 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \







echo "**************************"
echo "        template 1: arg"
echo "**************************"


# echo "=========================================================================================="
# echo "                                          real arg + trigger action                                      "
# echo "=========================================================================================="

# python code/run_args_qa_thresh.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --test_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --gold_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 3e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_output_thresh \
#   --nth_query 1 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \

echo "=========================================================================================="
echo "                                          real arg + trigger verb                                      "
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
  --output_dir args_qa_output_thresh \
  --nth_query 1 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \


echo "=========================================================================================="
echo "                                          real arg                                     "
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
  --output_dir args_qa_output_thresh \
  --nth_query 0 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \


echo "=========================================================================================="
echo "                                          gold arg + trigger                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
  --nth_query 1 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \

echo "=========================================================================================="
echo "                                          gold arg                                        "
echo "=========================================================================================="

python code/run_args_qa_thresh.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --gold_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 4e-5 \
  --num_train_epochs 6 \
  --output_dir args_qa_output_thresh \
  --nth_query 0 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --eval_test \
  --eval_per_epoch 20 \
  --max_seq_length 180 \
  --n_best_size 20 \
  --max_answer_length 3 \
  --larger_than_cls \



# echo "=========================================================================================="
# echo "                                           thresh real des_query + trigger action (squad)                                 "
# echo "=========================================================================================="

# python code/run_args_qa_thresh.py \
#   --do_train \
#   --do_eval \
#   --model squad2_output \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --test_file $ACE_PRE_DIR_ACTION/trigger_predictions.json \
#   --gold_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --output_dir args_qa_output_thresh \
#   --nth_query 5 \
#   --normal_file $ARG_QUERY_FILE \
#   --des_file $DES_QUERY_FILE \
#   --eval_test \
#   --eval_per_epoch 20 \
#   --max_seq_length 180 \
#   --n_best_size 20 \
#   --max_answer_length 3 \
#   --larger_than_cls \