Aspect Based Sentiment Analysis

# Model
## BERT-SPC
## AEN-BERT

# How to run?
## How to train a new model?

- data prepare

```
python data_prepare.py \
   --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
   --dataset=camera
```
或者直接执行 bash bin/run_dataprepare.sh

- train

```
python main.py \
    --do_eval \
    --do_train \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
    --num_train_epochs=20 \
    --save_checkpoints_steps=50 \
    --max_steps_without_decrease=500 \
    --batch_size=32 \
    --num_labels=1
```
或者直接执行 bash bin/run_main.sh

- test

```
python test.py --dataset=camera \
	--model_path=output/ckpts/export/best_exporter/1620294022 \
	--processor_file=output/preprocessor.json
```
或者直接执行 bash bin/run_test.sh

## How to run a serving client?
