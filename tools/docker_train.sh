GPUID=$1
CONFIG=$2

docker run -it --gpus all --ipc=host -v /disk3/yunseung/hbmap/:/data/ -v /disk3/seunghyeonkim/mmsegmentation/:/workspace/mmsegmentation/ --rm mmsegmentation:latest /bin/bash -c "cd /workspace/mmsegmentation; pip install -v -e .; CUDA_VISIBLE_DEVICES=${GPUID} python tools/train.py ${CONFIG}"