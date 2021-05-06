import os
import random
import json
import string
import opencc
import pandas as pd
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from bert import tokenization
from helper import DataProcessor, InputExample, file_based_convert_examples_to_features
from datasets import data_loader


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

flags = tf.flags
FLAGS = flags.FLAGS
FILE_HOME = os.path.abspath(os.path.dirname(__file__))

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "dataset", "camera",
    "dataset: camera | car | notebook | phone | all")

flags.DEFINE_string(
    "output_dir", os.path.join(FILE_HOME, "./output"),
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

def dump_to_tfrecords():
    if not tf.io.gfile.exists(os.path.join(FLAGS.output_dir, "tfrecords")):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, "tfrecords"))

    cc = opencc.OpenCC('t2s')
    ds = data_loader.ABSADataset(FLAGS.dataset)
    all_labels = ds.get_labels()

    data_processor = DataProcessor(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case,
        label_list=all_labels,
        max_seq_length=FLAGS.max_seq_length)
    data_processor.dump_to_file(os.path.join(FLAGS.output_dir, "preprocessor.json"))

    def create_examples(in_datas):
        examples = []
        for item in in_datas:
            content = item["text_left"] + item["aspect"] + item["text_right"]
            aspect = item['aspect']
            label = item['polarity']

            examples.append(
                InputExample(text_a=content, text_b=aspect, label=label))
        return examples
    
    train_examples = create_examples(ds.train_data)
    dev_examples = create_examples(ds.test_data)

    random.shuffle(train_examples)
    train_file = os.path.join(FLAGS.output_dir, "tfrecords/train.tf_record")
    file_based_convert_examples_to_features(train_examples, data_processor, train_file)
    
    dev_file = os.path.join(FLAGS.output_dir, "tfrecords/dev.tf_record")
    file_based_convert_examples_to_features(dev_examples, data_processor, dev_file)


if __name__ == "__main__":
    dump_to_tfrecords()
