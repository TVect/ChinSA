# -*- coding: utf-8 -*-

import time
import json
# import opencc
import tensorflow as tf
from helper import DataProcessor, InputExample


class ServingClient:

    def __init__(self, model_path, processor_file):
        self.preprocessor = DataProcessor.load_from_file(processor_file)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
        self.predict_fn = tf.contrib.predictor.from_saved_model(
            model_path, config=tf.ConfigProto(gpu_options=gpu_options))

    def predict(self, in_text, in_aspect):
        """
        @param in_text: string
        @param in_aspect: string
        """
        feature = self.preprocessor.convert_single_example(
            InputExample(text_a=in_text, text_b=in_aspect))
        output = self.predict_fn({"input_ids": [feature.input_ids], 
                                  "input_mask": [feature.input_mask], 
                                  "segment_ids": [feature.segment_ids]})
        return {"text": in_text, 
                "aspect": in_aspect, 
                "polarity_pos": output["polarity_pos_probs"][0].round(4).tolist(),
                "polarity_neg": output["polarity_neg_probs"][0].round(4).tolist()}


if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser(usage='"usage:%prog [options] arg1,arg2"')
    parser.add_option('-m', '--model_path', dest='model_path', action='store',
                      type=str, help='path to model: e.g. output/1560850786')
    parser.add_option('-p', '--processor_file', dest='processor_file', action='store',
                      type=str, help='path to processor_file: e.g. output/data_processor.json')
    options, args = parser.parse_args()
    
    client = ServingClient(model_path=options.model_path, 
                           processor_file=options.processor_file)
    in_text = "非常人性化的设计,很糟糕的手感"
    in_aspect = "设计"
    output = client.predict(in_text, in_aspect)
    print(output)

    in_aspect = "手感"
    output = client.predict(in_text, in_aspect)
    print(output)

