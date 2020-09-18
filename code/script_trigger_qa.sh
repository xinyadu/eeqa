#!/bin/sh

export ACE_DIR=./proc/data/ace-event/processed-data/json


echo "=========================================================================================="
echo "                                          query 5 'verb'                                  "
echo "=========================================================================================="

python code/run_trigger_qa.py \
  --do_infer \
  --do_eval \
  --eval_test \
  --model bert-base-uncased \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/dev_convert.json  \
  --test_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --eval_per_epoch 20 \
  --num_train_epochs 6 \
  --output_dir trigger_qa_output \
  --learning_rate 4e-5 \
  --nth_query 5 \
  --warmup_proportion 0.1 \
  
