#!/bin/bash

VOCAB_FILE="./chinese_L-12_H-768_A-12/vocab.txt"

DATASET="all"

python data_prepare.py --vocab_file=$VOCAB_FILE --dataset=$DATASET
