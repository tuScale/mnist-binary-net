import os
import argparse
import numpy as np

from consts import *

def binarize(x):
    return np.int8(2. * np.greater_equal(x, 0) - 1.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert MNIST parameters for x-intercept/sigma usage')
    parser.add_argument('--out', dest = 'dest_file_name', default = 'parameters.npz',
                        help = 'destination file')
    parser.add_argument('--xint_dt', dest = 'x_intercept_dt',
                        default = 'float32', choices = permitted_xint_dts,
                        help = 'x-intercept data type')
    args = parser.parse_args()

    # start out fresh
    if os.path.exists(args.dest_file_name):
      os.remove(args.dest_file_name)

    # Load original parameters
    # Outputted order is as follows:
    # Layer0    : W, xIntercept, sigmaSign
    # Layer1    : W, xIntercept, sigmaSign
    # Layer2    : W, xIntercept, sigmaSign
    # OutLayer  : W, b, beta, gamma, mu, inv_std
    print('Loading original parameters ...')
    f = np.load('../mnist_parameters.npz')
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    # start of by storing the configuration parameters
    bin_param_values = [[permitted_xint_dts.index(args.x_intercept_dt)]]

    # Load & compute input + hidden layer parameters
    x_intercept_caster = getattr(np, args.x_intercept_dt)
    binarizeFunc = np.vectorize(binarize)
    for i in range(3):
        print('Processing layer %d ...' % i)
        arr_id = i * 6
        w_orig = f['arr_%d' % arr_id]
        w = binarizeFunc(w_orig)
        x_intercept = []
        for i in range(len(f['arr_%d' % (arr_id + 1)])):
            bias = f['arr_%d' % (arr_id + 1)][i]
            beta = f['arr_%d' % (arr_id + 2)][i]
            gamma = f['arr_%d' % (arr_id + 3)][i]
            mean = f['arr_%d' % (arr_id + 4)][i]
            std = 1./f['arr_%d' % (arr_id + 5)][i]
            x_intercept.append(x_intercept_caster(mean - beta*std/gamma - bias));
        sigma_sign = [np.int8(-1. if x < 0 else 1.) for x in f['arr_%d' % (arr_id + 3)]]
        bin_param_values.extend([w, x_intercept, sigma_sign])

    # Load the last layer params
    print('Processing final layer ...')
    w_out = f['arr_18']
    b_out = f['arr_19']
    beta_out = f['arr_20']
    gamma_out = f['arr_21']
    mu_out = f['arr_22']
    inv_std_out = f['arr_23']
    bin_param_values.extend([w_out, b_out, beta_out, gamma_out, mu_out, inv_std_out])

    # save everything
    print('Saving everything ...')
    np.savez(args.dest_file_name, *bin_param_values)
