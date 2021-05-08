以下操作和效果都是在 P40 机器上执行的

**参考文档**

    https://github.com/NVIDIA/TensorRT/tree/20.10/demo/BERT

## 操作步骤

**克隆仓库**

```
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git checkout -b tag20.10 20.10
```

**启动容器**

```
nvidia-docker run --rm --privileged -it -v TensorRT:/workspace/TensorRT nvcr.io/nvidia/tensorrt:20.10-py3
```

准备 model checkpoint & engine 构建评测脚本

1. 拷贝 model checkpoint 到 /workspace/TensorRT/demo/BERT/models/absa_bert_spec

2. 将 engine 构建脚本 abas_builder.py 和 inference 脚本 absa_inference.py 放到 /workspace/TensorRT/demo/BERT/models/

3. 执行如下的 build engine 命令 和 inference 命令

**build engine**

```
python3 absa_builder.py -m /workspace/TensorRT/demo/BERT/models/absa_bert_spec/model.ckpt-1186 -o /workspace/TensorRT/demo/BERT/engines/absa_bert_spc_128.engine -b 1 -s 128 --fp32 -c /workspace/TensorRT/demo/BERT/models/absa_bert_spec/
```
**inference**

```
CUDA_VISIBLE_DEVICES="1" python3 absa_inference.py -e /workspace/TensorRT/demo/BERT/engines/absa_bert_spc_128.engine -v /workspace/TensorRT/demo/BERT/models/absa_bert_spec/vocab.txt -b 1
```

## 效果

**使用原始 Tensorflow Estimator Inference 的测试效果**

              qps  inference_time
count  100.000000      100.000000
mean    74.399650       13.530974
std      6.537310        1.042673
min     60.623594        9.899855
25%     70.919811       13.448834
50%     72.119747       13.865829
75%     74.355894       14.100432
max    101.011584       16.495228

**使用 TensorRT engine 的测试效果**

```
              qps  time_infer  time_gpu2cpu  time_memory  all_duration
count  100.000000  100.000000    100.000000   100.000000    100.000000
mean   219.798979    4.560747      0.020862     1.712787      6.693616
std      8.590433    0.287913      0.001946     0.647999      1.010695
min    134.986612    4.510880      0.019312     1.521826      6.366014
25%    220.257264    4.521847      0.020027     1.563370      6.427288
50%    220.805139    4.528880      0.020504     1.611471      6.479502
75%    221.148582    4.540145      0.020802     1.716912      6.658435
max    221.686258    7.408142      0.036240     8.042097     15.515327
```

