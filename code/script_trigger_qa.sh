#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json

# echo "=========================================================================================="
# echo "                                          query 4 action"
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_DIR/test_convert.json \
#   --test_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --eval_per_epoch 20 \
#   --num_train_epochs 8 \
#   --output_dir trigger_qa_action_output \
#   --learning_rate 4e-5 \
#   --nth_query 4 \
#   --warmup_proportion 0.1 \

echo "=========================================================================================="
echo "                                          query 5 verb"
echo "=========================================================================================="

python code/run_trigger_qa.py \
  --do_train \
  --do_eval \
  --eval_test \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/test_convert.json \
  --test_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --eval_per_epoch 20 \
  --num_train_epochs 8 \
  --output_dir trigger_qa_verb_old_output \
  --learning_rate 4e-5 \
  --nth_query 5 \
  --warmup_proportion 0.1 \
  
# echo "=========================================================================================="
# echo "                                          query 0 what is the trigger in the event?"
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_DIR/test_convert.json \
#   --test_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --eval_per_epoch 20 \
#   --num_train_epochs 8 \
#   --output_dir trigger_qa_output \
#   --learning_rate 4e-5 \
#   --nth_query 0 \


# echo "=========================================================================================="
# echo "                                          query 1 what happened in the event?"
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_DIR/test_convert.json \
#   --test_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --eval_per_epoch 20 \
#   --num_train_epochs 8 \
#   --output_dir trigger_qa_output \
#   --learning_rate 4e-5 \
#   --nth_query 1 \

# echo "=========================================================================================="
# echo "                                          query 2 trigger"
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_DIR/test_convert.json \
#   --test_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --eval_per_epoch 20 \
#   --num_train_epochs 8 \
#   --output_dir trigger_qa_output \
#   --learning_rate 4e-5 \
#   --nth_query 2 \




# echo "=========================================================================================="
# echo "                                          query 6 null"
# echo "=========================================================================================="

# python code/run_trigger_qa.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file $ACE_DIR/train_convert.json \
#   --dev_file $ACE_DIR/test_convert.json \
#   --test_file $ACE_DIR/test_convert.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --eval_per_epoch 20 \
#   --num_train_epochs 8 \
#   --output_dir trigger_qa_output \
#   --learning_rate 4e-5 \
#   --nth_query 6 \


