## Train
#### Requirements
python==3.6  
cuda==10.1    
cudnn==765    
mxnet-cu101==1.6.0.post0  
pip install easydict mxboard opencv-python tqdm    
[nccl](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)  
[openmpi](mxnet/setup-utils/install-mpi.sh)==4.0.0  
[horovod](mxnet/setup-utils/install-horovod.sh)==0.19.2  

#### Failures due to SSH issues
The host where horovodrun is executed must be able to SSH to all other hosts without any prompts.

#### Run with horovodrun
Typically one GPU will be allocated per process, so if a server has 8 GPUs, you will run 8 processes. 
In horovodrun, the number of processes is specified with the -np flag.

To run on a machine with 8 GPUs:
```shell script
horovodrun -np 8 -H localhost:8 bash config.sh
```

To run on two machine with 16 GPUs:
```shell script
horovodrun -np 16 -H ip1:8,ip2:8 bash config.sh
```

#### Run with mpi
```shell script
bash run.sh
```


## Troubleshooting

### Horovod installed successfully?  

Run `horovodrun --check` to check the installation of horovod.
```shell script
# Horovod v0.19.2:
# 
# Available Frameworks:
#     [ ] TensorFlow
#     [X] PyTorch
#     [X] MXNet
# 
# Available Controllers:
#     [X] MPI
#     [X] Gloo
# 
# Available Tensor Operations:
#     [X] NCCL
#     [ ] DDL
#     [ ] CCL
#     [X] MPI
#     [X] Gloo
```

### Mxnet Version!
Some versions of mxnet with horovod have bug.   
It is recommended to try version **1.5 or 1.6**.

**The community has found that mxnet1.5.1 cannot install horovod.**

### Check CUDA version!
```shell script
# Make sure your cuda version is same as mxnet, such as mxnet-cu101 (CUDA 10.1)

/usr/local/cuda/bin/nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2019 NVIDIA Corporation
# Built on Wed_Apr_24_19:10:27_PDT_2019
# Cuda compilation tools, release 10.1, V10.1.168
```

### Block IO
You can turn on the debug mode to check whether your slow training speed is the cause of IO.

### Training Speed.
If you find that your training speed is the io bottleneck, you can mount dataset to RAM, 
using the following command.
```shell script
# If your RAM has 256G
sudo mkdir /train_tmp
mount -t tmpfs -o size=140G  tmpfs /train_tmp
```

# Pretrain Test
```

[[ 2.4450580e-03 -3.6629855e-03 -5.4821405e-03 ...  1.7617886e-03
  -1.7111700e-02 -1.2869146e-04]
 [-4.0940483e-04 -5.2541954e-04 -3.5010649e-05 ...  3.8271716e-03
   2.9155412e-03 -2.8612907e-03]
 [-5.4146082e-04  8.5490878e-04  2.6085819e-03 ... -1.0771470e-03
   1.6705096e-02  8.4809829e-03]
 ...
 [-9.8874152e-04 -8.9934692e-03  9.2194683e-04 ...  2.4383611e-03
   5.3678402e-03 -3.4479324e-03]
 [-2.9350563e-03 -1.1713833e-02 -8.6075356e-03 ... -3.1413541e-03
   2.2503075e-03  1.0103217e-02]
 [ 9.6949777e-03 -1.5208530e-03  3.0143571e-03 ...  3.1245875e-03
  -5.2550877e-03  2.3517154e-05]]
<NDArray 45029x512 @gpu(7)>]
```
