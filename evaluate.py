
import sys
import os
import time
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_ops
import dataset

if __name__ == "__main__":

    batch_size = 10000
    print("batch_size = "+str(batch_size))

    # MLP parameters
    num_units = 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))

    # kernel = "baseline"
    # kernel = "xnor"
    kernel = "theano"
    print("kernel = "+ kernel)

    print('Loading MNIST dataset...')
    _, _2, test_set = dataset.load_dataset(flatten_x = True, one_hot = False, scale_y = False)

    print('Building the MLP...')

    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.vector('targets')

    mlp = lasagne.layers.InputLayer(shape=(None, 784),input_var=input)

    # Input layer is not binary -> use baseline kernel in first hidden layer
    mlp = binary_ops.DenseLayer(
            mlp,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=num_units,
            kernel = kernel)

    mlp = lasagne.layers.BatchNormLayer(mlp)
    mlp = lasagne.layers.NonlinearityLayer(mlp,nonlinearity=binary_ops.SignTheano)

    for k in range(1,n_hidden_layers):

        mlp = binary_ops.DenseLayer(
                mlp,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units,
                kernel = kernel)

        mlp = lasagne.layers.BatchNormLayer(mlp)
        mlp = lasagne.layers.NonlinearityLayer(mlp,nonlinearity=binary_ops.SignTheano)

    mlp = binary_ops.DenseLayer(
                mlp,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10,
                kernel = kernel)

    mlp = lasagne.layers.BatchNormLayer(mlp)
    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_output_class = T.argmax(test_output, axis=1)
    test_err = T.mean(T.neq(test_output_class, target),dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input], test_output_class)
    val_fn_error = theano.function([input, target], test_err)

    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    with np.load('mnist_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(mlp, param_values)

    # Binarize the weights
    weights_count = 0
    params = lasagne.layers.get_all_params(mlp)
    for param in params:
        print (param.name)
        if param.name == "W":
            orig_weights = param.get_value()
            param.set_value(binary_ops.SignNumpy(orig_weights))
            weights_count = weights_count + len(orig_weights.reshape(-1))
        print (param.get_value())

    # Print some stats
    total_params_count = lasagne.layers.count_params(mlp)
    weights_needed_bytes_count = weights_count/8
    non_weights_needed_bytes_count = total_params_count * 4 - weights_count / 8
    print('The model has a total of %d parameters of which %d (the weights) are binary.' % (total_params_count, weights_count))
    print('Space requirements:')
    print('  - %d bytes for the weights' % weights_needed_bytes_count)
    print('  - %d bytes for the other parameters (assumed 32bit floats each)' % non_weights_needed_bytes_count)
    print('  = %d bytes' % (weights_needed_bytes_count + non_weights_needed_bytes_count))

    print('Running tests...')

    test_subsampled_indeces = np.random.choice(len(test_set['y']), 10)
    test_out = val_fn(test_set['X'][test_subsampled_indeces])
    print ('Predicted output for 10 randomly sampled digits: %s' % test_out)
    print ('Expected values were: %s' % test_set['y'][test_subsampled_indeces])

    start_time = time.time()
    test_error = val_fn_error(test_set['X'],test_set['y'])*100.
    run_time = (time.time() - start_time) * 1000

    print("Final test error: %.5f%%" % test_error)
    print("Time required to run all tests: %.4fms" % run_time)

