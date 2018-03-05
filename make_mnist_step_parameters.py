import sys
import numpy as np

def binarize(x):
    return np.float32(2. * np.greater_equal(x, 0) - 1.)

if __name__ == "__main__":
    # Load original parameters
    # Outputted order is as follows:
    # Layer0    : W, xIntercept, sigmaSign
    # Layer1    : W, xIntercept, sigmaSign
    # Layer2    : W, xIntercept, sigmaSign
    # OutLayer  : W, b, beta, gamma, mu, inv_std
    print('Loading original parameters ...')
    f = np.load('mnist_parameters.npz')
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    bin_param_values = []

    # Load & compute input + hidden layer parameters
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
            x_intercept.append(np.float32(mean - beta*std/gamma - bias));
        sigma_sign = [np.float32(-1. if x < 0 else 1.) for x in f['arr_%d' % (arr_id + 3)]]
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
    np.savez('mnist_step_parameters.npz', *bin_param_values)
