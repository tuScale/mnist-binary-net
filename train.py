import sys
import os
import time
from collections import OrderedDict

import numpy as np
# Uncomment the following line to reach article performance specs
# np.random.seed(1234)

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net
import dataset

if __name__ == "__main__":
    print ('\nNetwork parameters')
    batch_size = 100
    num_units = 4096
    n_hidden_layers = 3
    alpha = .1 # alpha is the exponential moving average factor
    epsilon = 1e-4

    print('  %-32s %d' % ('batch size', batch_size))
    print('  %-32s %.5f' % ('alpha', alpha))
    print('  %-32s %.5f' % ('epsilon', epsilon))
    print('  %-32s %d' % ('number of neurons per layer', num_units))
    print('  %-32s %d' % ('number of hidden layers', n_hidden_layers))

    print ('\nTraining parameters')
    save_path = "mnist_parameters.npz"
    num_epochs = 1000
    LR_start = .003
    LR_fin = 0.0000003
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    shuffle_parts = 1
    dropout_in = .2 # Dropout parameters
    dropout_hidden = .5
    activation = binary_net.binary_tanh_unit
    binary = True
    stochastic = False
    H = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper

    print('  %-32s %d' % ('number of epochs', num_epochs))
    print('  %-32s %.2f' % ('inward dropout ratio', dropout_in))
    print('  %-32s %.2f' % ('outward dropout ratio', dropout_hidden))
    print('  %-32s %s' % ('activation used by binary nodes', 'binary_net.binary_tanh_unit'))
    print('  %-32s %s' % ('work in binary mode', str(binary)))
    print('  %-32s %s' % ('operate stochasticly', str(stochastic)))
    print('  %-32s %s' % ('possible weight values', '[-%d, +%d]' % (H, H)))
    print('  %-32s %s' % ('W_LR_scale', str(W_LR_scale)))
    print('  %-32s %.7f' % ('learning rate start value', LR_start))
    print('  %-32s %.7f' % ('learning rate target value', LR_fin))
    print('  %-32s %.7f' % ('learning rate decay rate', LR_decay))
    print('  %-32s %s' % ('model save file name', save_path))
    print('  %-32s %d' % ('shuffled parts', shuffle_parts))

    print('\nLoading the MNIST dataset...')
    train_set, valid_set, test_set = dataset.load_dataset()

    print('\nBuilding the multi-layer perceptron (MLP) network...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    # Input layer
    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)
    mlp = lasagne.layers.DropoutLayer(
            mlp,
            p=dropout_in)

    # Hidden layers
    for k in range(n_hidden_layers):
        mlp = binary_net.DenseLayer(
                mlp,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon,
                alpha=alpha)
        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
        mlp = lasagne.layers.DropoutLayer(
                mlp,
                p=dropout_hidden)

    # Final layer
    mlp = binary_net.DenseLayer(
                mlp,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)

    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon,
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))

    if binary:
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)

        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training the network ...')
    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set['X'],train_set['y'],
            valid_set['X'],valid_set['y'],
            test_set['X'],test_set['y'],
            save_path,
            shuffle_parts)
