#!/bin/bash

BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
# BERT_BASE_DIR="/home/xiaoliang.qxl/DiskHome/bert/tmp/pretraining_output_1m"
# BERT_BASE_DIR="/home/xiaoliang.qxl/DiskHome/bert/tmp3/pretraining_output_1.5m"
# BERT_BASE_DIR="/home/xiaoliang.qxl/DiskHome/bert/tmp/pretraining_output_1.5m"


python main.py \
    --do_eval \
    --do_train \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --num_train_epochs=20 \
    --save_checkpoints_steps=50 \
    --max_steps_without_decrease=500 \
    --batch_size=32 \
    --num_labels=2
