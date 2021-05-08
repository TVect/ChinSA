#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
import os

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

from helper import DataProcessor, InputExample


class TritonClient:

    def __init__(self, url):
        try:
            concurrency = 1
            self.client = httpclient.InferenceServerClient(
                url=url, concurrency=concurrency)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

    def inference(self, batch_inputs, model_name="bert_spc_tensorrt"):
        inputs, outputs = [], []

        inputs.append(httpclient.InferInput('input_ids', [1, 128], "INT32"))
        inputs.append(httpclient.InferInput('input_mask', [1, 128], "INT32"))
        inputs.append(httpclient.InferInput('segment_ids', [1, 128], "INT32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(batch_inputs["input_ids"], binary_data=True)
        inputs[1].set_data_from_numpy(batch_inputs["input_mask"], binary_data=True)
        inputs[2].set_data_from_numpy(batch_inputs["segment_ids"], binary_data=True)

        # outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
        outputs.append(httpclient.InferRequestedOutput('output/probs', binary_data=False))
        results = self.client.infer(model_name, inputs, outputs=outputs)
        return results


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import time

    triton_client = TritonClient(url="10.58.11.32:8000")
    batch_inputs = {
            "input_ids": np.ones(shape=[1, 128], dtype=np.int32), 
            "input_mask": np.ones(shape=[1, 128], dtype=np.int32), 
            "segment_ids": np.ones(shape=[1, 128], dtype=np.int32)}
    result = triton_client.inference(batch_inputs, model_name="bert_spc_tf")
    print("result:", result.get_response())

    durations = []
    for _ in range(100):
        start_time = time.time()
        result = triton_client.inference(batch_inputs, model_name="bert_spc_tf")
        result.get_response()
        durations.append((time.time() - start_time) * 1000)
    print(pd.DataFrame({"inference time": durations}).describe())
