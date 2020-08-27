#!/bin/sh

export SQUAD_DIR=../squad_data

# echo "=========================================================================================="
# echo "                                          squad start max length 512, lr 1e-6                                  "
# echo "=========================================================================================="

# python code/run_squad_start.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/train-v2.0.json \
#   --dev_file $SQUAD_DIR/dev-v2.0.json \
#   --test_file $SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 1e-6 \
#   --num_train_epochs 2 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --eval_metric f1 \
#   --output_dir squad2_output_start \
#   --version_2_with_negative \

# echo "=========================================================================================="
# echo "                                          squad start max length 384, lr 1e-6                                  "
# echo "=========================================================================================="

# python code/run_squad_start.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/train-v2.0.json \
#   --dev_file $SQUAD_DIR/dev-v2.0.json \
#   --test_file $SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 1e-6 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --eval_metric f1 \
#   --output_dir squad2_output_start_384 \
#   --version_2_with_negative \

# echo "=========================================================================================="
# echo "                                          squad start max length 128, lr 1e-6                                  "
# echo "=========================================================================================="

# python code/run_squad_start.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/train-v2.0.json \
#   --dev_file $SQUAD_DIR/dev-v2.0.json \
#   --test_file $SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size 32 \
#   --eval_batch_size 32  \
#   --learning_rate 1e-6 \
#   --num_train_epochs 2 \
#   --max_seq_length 128 \
#   --doc_stride 32 \
#   --eval_metric f1 \
#   --output_dir squad2_output_start_128 \
#   --version_2_with_negative \

echo "=========================================================================================="
echo "                                          squad max length 20, lr 2e-5                                            "
echo "=========================================================================================="

python code/run_squad.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --dev_file $SQUAD_DIR/dev-v2.0.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 20 \
  --doc_stride 10 \
  --eval_metric best_f1 \
  --output_dir squad2_output_20 \
  --version_2_with_negative \
  # --fp16

echo "=========================================================================================="
echo "                                          squad max length 40, lr 2e-5"
echo "=========================================================================================="


python code/run_squad.py \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --dev_file $SQUAD_DIR/dev-v2.0.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 40 \
  --doc_stride 20 \
  --eval_metric best_f1 \
  --output_dir squad2_output_40 \
  --version_2_with_negative \
  # --fp16


# echo "=========================================================================================="
# echo "                                          squad max length 512, lr 2e-5"
# echo "=========================================================================================="


# python code/run_squad.py \
#   --do_train \
#   --do_eval \
#   --model bert-base-uncased \
#   --train_file $SQUAD_DIR/train-v2.0.json \
#   --dev_file $SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size 8 \
#   --eval_batch_size 8  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 1 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --eval_metric best_f1 \
#   --output_dir squad2_output \
#   --version_2_with_negative \

# python run_squad.py \
#   --bert_model bert-base-uncased \
#   --do_train \
#   --do_predict \
#   --version_2_with_negative \
#   --do_lower_case \
#   --train_file $SQUAD_DIR/sample_train-v2.0.json \
#   --predict_file $SQUAD_DIR/sample_dev-v2.0.json \
#   --train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2.0 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir /tmp/debug_squad/