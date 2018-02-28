
import time
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

import lasagne

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))

# The weights' binarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        Wb = W
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)

        # Stochastic BinaryConnect
        if stochastic:
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            Wb = T.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)

    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    def __init__(self, incoming, num_units,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights
            self.params[self.W]=set(['binary'])
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        self.W = Wr

        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        self.binary = binary
        self.stochastic = stochastic
        self.H = H

        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)

    def convolve(self, input, deterministic=False, **kwargs):
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        self.W = Wr

        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:
        params = layer.get_params(binary=True)
        if params:
            grads.append(theano.grad(loss, wrt=layer.Wb))

    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)

    for layer in layers:
        params = layer.get_params(binary=True)
        for param in params:
            print('  %-32s %.6f' % ('W_LR_scale', layer.W_LR_scale))
            print('  %-32s %.6f' % ('H', layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)

    return updates

# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1):

    # A function which shuffles a dataset
    def shuffle(X,y):
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])

        for k in range(shuffle_parts):
            np.random.shuffle(shuffled_range)
            for i in range(chunk_size):
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer

        return X,y

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
        loss/=batches

        return loss

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        err = 0
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss

        err = err / batches * 100
        loss /= batches

        return err, loss

    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        val_err, val_loss = val_epoch(X_val,y_val)

        # test if validation error went down
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = epoch+1
            test_err, test_loss = val_epoch(X_test,y_test)

            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch %d/%d took %.5fs" % (epoch+1, num_epochs, epoch_duration))
        print("  %-32s %.6f" % ("Learning rate", LR))
        print("  %-32s %.6f" % ("training loss", train_loss))
        print("  %-32s %.6f" % ("validation loss", val_loss))
        print("  %-32s %.6f%%" % ("validation error rate", val_err))
        print("  %-32s %d" % ("best epoch", best_epoch))
        print("  %-32s %.6f%%" % ("best validation error rate", best_val_err))
        print("  %-32s %.6f" % ("test loss", test_loss))
        print("  %-32s %.6f%%" %("test error rate", test_err))

        # decay the LR
        LR *= LR_decay
