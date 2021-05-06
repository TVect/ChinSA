#!/bin/bash

MODEL_PATH="output/ckpts/export/best_exporter/1620294022/"
PROCESSOR_FILE="output/data_processor.json"
DATASET="camera"

python test.py --dataset=$DATASET \
	--model_path=$MODEL_PATH \
	--processor_file=$PROCESSOR_FILE

