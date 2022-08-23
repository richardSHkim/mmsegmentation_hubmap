## Introduction

Codes for [HuBMAP Challenge](https://www.kaggle.com/competitions/hubmap-organ-segmentation) based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

## Usage

* build docker image
```bash
cd docker
docker build -t mmsegmentation .
```

* edit `tools/docker_train.sh` file

Repalce [DIR_TO_DATA] and [DIR_TO_REPO].

```shell
GPUID=$1
CONFIG=$2

docker run -it --gpus all --ipc=host \
  -v [DIR_TO_DATA]:/data/ \
  -v [DIR_TO_REPO]:/workspace/mmsegmentation/ \
  --rm mmsegmentation:latest \
  /bin/bash -c \
  "cd /workspace/mmsegmentation; pip install -v -e .; CUDA_VISIBLE_DEVICES=${GPUID} python tools/train.py ${CONFIG}"
```

* train
```bash
sh tools/docker_train.sh [GPU_ID] [CONFIG_FILE]
```