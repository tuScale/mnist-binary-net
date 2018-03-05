
import sys
import os
import time

import numpy as np

import theano
import theano.tensor as T
import lasagne

import dataset

class BinaryDenseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W = lasagne.init.Normal(0.01), xInt = lasagne.init.Normal(0.01), sigmaSgn = lasagne.init.Normal(0.01), **kwargs):
        super(BinaryDenseLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.xIntercept = self.add_param(xInt, (num_units, ), name='xIntercept')
        self.sigmaSign = self.add_param(sigmaSgn, (num_units, ), name='sigmaSign')

    def _non_liniarity(self, x):
        # Since the activation function is binary, we only care about the sign of the squashed batch-norm
        # comparison.
        # This basically implements the x-intercept step operation:
        #     neg(sign(sigma)) if x < mu - (beta * std/gamma) - bias else sign(sigma)
        return (2 * T.gt(x - self.xIntercept, 0) - 1.) * self.sigmaSign

    def get_output_for(self, input, **kwargs):
        # TODO: XNORify this
        xW = T.dot(input, self.W)
        return self._non_liniarity(xW)

    def get_output_shape_for(self, input_shape):
        output_shape = (input_shape[1], self.num_units)
        return output_shape

if __name__ == "__main__":
    # MLP parameters
    num_units = 4096
    n_hidden_layers = 3
    print("Number of hidden units  : %-6d" % num_units)
    print("Number of hidden layers : %-6d" % n_hidden_layers)

    print('Loading MNIST dataset...')
    _, _2, test_set = dataset.load_dataset(flatten_x = True, one_hot = False, scale_y = False)

    print('Building the MLP...')
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.vector('targets')

    # Make the actual network
    mlp = lasagne.layers.InputLayer(shape=(None, 784),input_var=input)
    for k in range(n_hidden_layers):
        mlp = BinaryDenseLayer(mlp, num_units = num_units)
    mlp = lasagne.layers.DenseLayer(
                mlp,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)
    mlp = lasagne.layers.BatchNormLayer(mlp)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_output_class = T.argmax(test_output, axis=1)
    test_err = T.mean(T.neq(test_output_class, target), dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input], test_output_class)
    val_fn_error = theano.function([input, target], test_err)

    # Load the network parameters
    print("Loading the trained parameters and binarizing the weights...")
    with np.load('mnist_step_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(mlp, param_values)

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

