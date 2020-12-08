#! /bin/bash
HOROVOD_CUDA_HOME=~/app/cuda-10.1/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod==0.19.2

