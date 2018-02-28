import sys
import os
import time

import numpy as np

from urllib import urlretrieve
import gzip

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays

def load_dataset(flatten_x = False, one_hot = True, scale_y = True):
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename, flatten = False):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        # or, if a flatten result is requested: (examples, channels * rows * columns)
        if flatten:
            data = data.reshape(-1, 784)
        else:
            data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        data = data / np.float32(255)
        # Finally, we translate the values in the [-1, +1] domain
        return 2 * data - 1.

    def load_mnist_labels(filename, one_hot = True, scale = True):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, one-hot encode them
        if one_hot:
            data = np.float32(np.eye(10)[data])
            # And translate them to [-1, +1] domain as well
            if scale:
               data = 2 * data - 1.
        return data

    # flatten targets
    # train_set.y = np.hstack(train_set.y)
    # valid_set.y = np.hstack(valid_set.y)
    # test_set.y = np.hstack(test_set.y)

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz', flatten_x)
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz', one_hot, scale_y)
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz', flatten_x)
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz', one_hot, scale_y)

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order
    return {'X': X_train, 'y': y_train}, {'X': X_val, 'y': y_val}, {'X': X_test, 'y': y_test}

