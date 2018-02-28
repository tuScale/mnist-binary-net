#! /bin/bash

ENV_NAME=mnist-binary-net
CUDA_ROOT=/usr/local/cuda

export PATH=$CUDA_ROOT/bin:$PATH
export CUDA_INC_DIR=$CUDA_ROOT/include

# create a conda environment with the following packages installed:
#  -> mkl-service v1.1.2
#  -> nose v1.3.7
#  -> numpy v1.14.1
#  -> python v2.7.14
#  -> pygpu v0.7.5
# + theano, pycuda and lasagne
conda create -n $ENV_NAME -q mkl-service=1.1.2 nose=1.3.7 numpy=1.14.1 python=2.7.14 pygpu=0.7.5 &&
source activate $ENV_NAME &&
pip install pycuda &&
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip &&
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip &&
echo -e "\nEverything set. Type 'source activate $ENV_NAME' to use it.\n"
