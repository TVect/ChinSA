**convert savedmodel to pb**

- 查看 savedmodel 信息

```
saved_model_cli show --dir=savedmodel_dir --all
```

- 模型文件转换

```
freeze_graph --input_saved_model_dir=savedmodel_dir --output_node_names=output/probs --output_graph=freeze_graph.pb
```

- 准备推理文件

1. tensorflow backend 可以使用 xxxxx.pb 做为 model.graphdef 放在 bert_spc_tf 中

2. tensorrt backend 可以使用 xxxxx.engine 做为 model.plan 放在 bert_spc_tensorrt 中


- 启动 triton inference server

```
nvidia-docker run -e CUDA_VISIBLE_DEVICES=3 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path_to_project/inference/triton_bert_spc/model-repository:/models nvcr.io/nvidia/tritonserver:20.10-py3 tritonserver --model-repository=/models
```

参考自：https://github.com/triton-inference-server/server/blob/r20.10/docs/quickstart.md

## 效果

- tensorflow backend + tensorrt optimization

```
       inference time
count      100.000000
mean         9.696705
std          0.944969
min          9.095669
25%          9.315670
50%          9.534121
75%          9.757757
max         17.907619
```

- tensorrt backend

```
       inference time
count      100.000000
mean         6.042502
std          0.047585
min          5.923033
25%          6.012201
50%          6.030798
75%          6.070852
max          6.252050
```