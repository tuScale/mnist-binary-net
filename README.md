# About
This is an updated version of the original [binary-net](https://github.com/MatthieuCourbariaux/BinaryNet) code release.
It only includes the mnist example, but it should be easier to setup, train and evaluate. 
I don't take any credit for the original work nor am I affiliated in any way with the paper authors. I just took
parts of their code base and updated it to so that it can be checked faster.

# Prerequisites
I managed to run the project with the following resources:
* An NVIDIA GPU PC (mine was a 970GTX, but any CUDA-enabled card should do)
* Ubuntu (I used 16.04)
* [Cuda SDK](https://developer.nvidia.com/cuda-downloads) (I tested it against v8.0 but v9.0 appears to be fine as well)
* [cuDNN](https://developer.nvidia.com/cudnn) 
* [Python's Conda Environment Manager](https://conda.io/docs/user-guide/install/linux.html#install-linux-silent)

# Setup
Run
```
$ ./setup.sh
```
it will create a new ```conda``` virtual environment with all the dependencies installed to be able to run the code.
To activate the environment just type
```
$ source activate mnist-binary-net
```

## Setting up Theano to use the GPU
In order to use the GPU to speed up the training/inference, make sure Theano is [aware of this](http://deeplearning.net/software/theano/library/config.html). One easy way to do this is by using a ```.theanorc``` file situated in the home directory. Once this created, just put the following lines inside it and you are ready to go:
```
[global]
floatX = float32
device = cuda0
```

# Train
This is the hardest part which will take a couple of hours to do (it took me arround 6 to run all those 1000 epochs). To do
this, type:
```
$ MKL_THREADING_LAYER=GNU python train.py
```
or, if you don't want to wait, you can download the pre-trained weights from [here](https://www.dropbox.com/s/q4djs5glajolnw4/mnist_parameters.npz?dl=0) [cca 140MB]. 
Once finished, just make sure the ```npz``` file is situated in the project folder if you want to evaluate it.

# Evaluate
To test the binary-net, just type
```
$ MKL_THREADING_LAYER=GNU python evaluate.py
```
you should see a bunch of output/statistics like:
  *  the restored network parameters (weights being binarized to [-1, +1])
  *  parameters space statistics
  *  a test-run with 10 randomly picked samples from the testing set along with their output
  *  the total cumulated error after evaluating the network against all the 10000 digits

Part of the output:
```
W
[[ 1. -1.  1. ... -1. -1. -1.]
 [ 1.  1. -1. ... -1.  1.  1.]
 [-1.  1.  1. ...  1.  1. -1.]
 ...
 [-1. -1.  1. ...  1.  1. -1.]
 [ 1.  1.  1. ... -1.  1.  1.]
 [-1. -1. -1. ... -1. -1.  1.]]
b
[-2.3605786e-05 -2.0557101e-05  4.4214099e-05  7.2509097e-06
 -4.5804554e-05  1.5261756e-05  2.5246634e-05  3.1179152e-06
 -5.8647605e-05  6.8424721e-05]
beta
[-1.8730884 -1.5041908 -1.59434   -1.6025535 -1.7137499 -1.8657532
 -1.9541619 -1.7874093 -1.627327  -1.6735605]
gamma
[1.649736  1.4843471 1.4078758 1.3849556 1.4668953 1.5560124 1.6956487
 1.562617  1.3272513 1.3941664]
mean
[ 67.91481   68.45975   46.644485 158.55453  176.3165   134.5987
  56.28972   31.798172 287.57718  276.70938 ]
inv_std
[0.00236199 0.00228287 0.00256304 0.0027011  0.00240227 0.00234367
 0.00206922 0.0021205  0.00274556 0.00246587]
The model has a total of 36868146 parameters of which 36806656 (the weights) are binary.
Space requirements:
  - 4600832 bytes for the weights
  - 142871752 bytes for the other parameters (assumed 32bit floats each)
  = 147472584 bytes
Running tests...
Predicted output for 10 randomly sampled digits: [8 0 8 2 1 2 4 7 6 6]
Expected values were: [8 0 8 2 1 2 4 7 6 6]
Final test error: 0.91000%
Time required to run all tests: 232.6229m
```
# Happy coding :beers:

# Notice
Please don't change the ```kernel``` in ```evaluate.py```. For now, it only works with theano's cpu matrix multiplcations.
GPU/XNOR version will be available afterwards.
