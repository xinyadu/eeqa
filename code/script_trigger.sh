#!/bin/sh

export ACE_DIR=./dygiepp/data/ace-event/processed-data/json

echo "=========================================================================================="
echo "                                          3e-5                                            "
echo "=========================================================================================="

python code/run_trigger.py \
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
  --output_dir trigger_output \
  --learning_rate 3e-5 \



echo "=========================================================================================="
echo "                                          2e-5                                            "
echo "=========================================================================================="

python code/run_trigger.py \
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
  --output_dir trigger_output \
  --learning_rate 2e-5 \
