#!/bin/sh

export ACE_DIR=./proc/data/ace-event/processed-data/json


echo "=========================================================================================="
echo "                                          query 5                                         "
echo "=========================================================================================="

python code/run_trigger_qa.py \
  --train_file $ACE_DIR/toy.json \
  --dev_file $ACE_DIR/toy.json  \
  --test_file $ACE_DIR/toy.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --eval_per_epoch 20 \
  --num_train_epochs 6 \
  --output_dir trigger_qa_output \
  --model_dir trigger_qa_output/epoch0-step0 \
  --learning_rate 4e-5 \
  --nth_query 5 \
  --warmup_proportion 0.1 \
  --do_train \
  --do_eval \
  --model bert-base-uncased 
  # you should (1) add eval_test; (2) set --model to your real model path for test (e.g., epoch-x-step-x)