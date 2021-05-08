#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script uses a prebuilt TensorRT BERT QA Engine to answer a question 
based on the provided passage. It additionally includes an interactive mode 
where multiple questions can be asked.
"""

import time
import json
import ctypes
import argparse
import collections
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import helpers.tokenization as tokenization
import helpers.data_processing as dp

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BERT QA Inference')
    parser.add_argument('-e', '--engine', dest='engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument('-i', '--in_text', nargs='*', dest='in_text',
            help='Text for classification',
            default='')
    parser.add_argument('-v', '--vocab_file', dest='vocab_file',
            help='Path to file containing entire understandable vocab',
            default='./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt')
    parser.add_argument('-b', '--batch_size', dest='batch_size',
            help='Batch size for inference', default=1, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    in_text = ""
    if not args.in_text == '':
        in_text = args.in_text[0]
    print("\nin_text: {}".format(in_text))

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
    max_query_length = 64
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride = 128
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    # max_seq_length = 384
    max_seq_length = 128
    # Extract tokecs from the paragraph
    # doc_tokens = dp.convert_doc_tokens(in_text)

    def text_features(text_a, text_b):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        # truncate_seq_pair
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_seq_length - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        tokens = ["[CLS]"]
        segment_ids = [0]
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        batch_input_ids = np.asarray([input_ids], dtype=np.int32, order=None)
        batch_input_mask = np.asarray([input_mask], dtype=np.int32, order=None)
        batch_segment_ids = np.asarray([segment_ids], dtype=np.int32, order=None)
        return {"input_ids": batch_input_ids, 
                "input_mask": batch_input_mask, 
                "segment_ids": batch_segment_ids}

    # Import necessary plugins for BERT TensorRT
    handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    if not handle:
        raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        # select engine profile
        selected_profile = -1
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
        for idx in range(engine.num_optimization_profiles):
            profile_shape = engine.get_profile_shape(profile_index = idx, binding = idx * num_binding_per_profile)
            # if profile_shape[0][1] <= args.batch_size and profile_shape[2][1] >= args.batch_size and profile_shape[0][0] <= max_seq_length and profile_shape[2][0] >= max_seq_length:
            #     selected_profile = idx
            #     break
            print("profile_shape:", profile_shape)
            if profile_shape[0][1] <= max_seq_length and profile_shape[2][1] >= max_seq_length and profile_shape[0][0] <= args.batch_size and profile_shape[2][0] >= args.batch_size:
                selected_profile = idx
                break

        if selected_profile == -1:
            raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))

        context.active_optimization_profile = selected_profile
        binding_idx_offset = selected_profile * num_binding_per_profile

        # Specify input shapes. These must be within the min/max bounds of the active profile 
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        # input_shape = (max_seq_length, args.batch_size)
        input_shape = (args.batch_size, max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
        for binding in range(3):
            context.set_binding_shape(binding_idx_offset + binding, input_shape)
        assert context.all_binding_shapes_specified

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # def inference(features, tokens):
        def inference(input_features):
            global h_output

            # _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
            #         "NetworkOutput",
            #         ["start_logits", "end_logits", "feature_index"])
            networkOutputs = []

            eval_time_elapsed = 0
            time_gpu2cpu = 0
            time_memory = 0
            # for input_feature in input_features:
            for idx in range(len(input_features["input_ids"])):
                # Copy inputs
                starttime_memory = time.time()
                input_ids_batch = np.dstack(
                    input_features["input_ids"][idx] * args.batch_size).squeeze()
                segment_ids_batch = np.dstack(
                    input_features["segment_ids"][idx] * args.batch_size).squeeze()
                input_mask_batch = np.dstack(
                    input_features["input_mask"][idx] * args.batch_size).squeeze()

                input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids_batch.ravel()))
                segment_ids = cuda.register_host_memory(np.ascontiguousarray(segment_ids_batch.ravel()))
                input_mask = cuda.register_host_memory(np.ascontiguousarray(input_mask_batch.ravel()))
                time_memory += time.time() - starttime_memory

                eval_start_time = time.time()
                cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
                cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
                cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

                # Run inference
                context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] + [int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
                # Synchronize the stream
                stream.synchronize()
                eval_time_elapsed += (time.time() - eval_start_time)

                # Transfer predictions back from GPU
                starttime_gpu2cpu = time.time()
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()

                # Only retrieve and post-process the first batch
                batch = h_output[0]
                time_gpu2cpu += (time.time() - starttime_gpu2cpu)
                # networkOutputs.append(_NetworkOutput(
                #     start_logits = np.array(batch.squeeze()[:, 0]),
                #     end_logits = np.array(batch.squeeze()[:, 1]),
                #     feature_index = feature_index
                #     ))

            eval_time_elapsed /= len(input_features["input_ids"])

            # Total number of n-best predictions to generate in the nbest_predictions.json output file
            n_best_size = 20

            # The maximum length of an answer that can be generated. This is needed
            # because the start and end predictions are not conditioned on one another
            max_answer_length = 30

            # print("====== h_output:", h_output[0])
            # prediction, nbest_json, scores_diff_json = dp.get_predictions(tokens, features,
            #         networkOutputs, args.n_best_size, args.max_answer_length)

            eval_time_elapsed *= 1000
            qps = args.batch_size /eval_time_elapsed * 1000

            # print("------------------------")
            # print("Running inference in {:.3f} Sentences/Sec ({:.3f}ms)".format(
            #     qps, eval_time_elapsed))
            # print("------------------------")

            return qps, eval_time_elapsed, time_gpu2cpu * 1000, time_memory * 1000


        import pandas as pd
        qps_list, time_infer_list, time_gpu2cpu_list, time_memory_list = [], [], [], []
        text_a = "非常人性化的设计啊"
        text_b = "设计"

        all_durations = []
        for _ in range(100):
            features = text_features(text_a, text_b)
            start_duration_time = time.time()
            qps, time_infer, time_gpu2cpu, time_memory = inference(features)
            all_durations.append((time.time() - start_duration_time) * 1000)
            qps_list.append(qps)
            time_infer_list.append(time_infer)
            time_gpu2cpu_list.append(time_gpu2cpu)
            time_memory_list.append(time_memory)
        print(pd.DataFrame(
            {"qps": qps_list, 
            "time_infer": time_infer_list,
            "time_gpu2cpu": time_gpu2cpu_list, 
            "time_memory": time_memory_list,
            "all_duration": all_durations}).describe())
