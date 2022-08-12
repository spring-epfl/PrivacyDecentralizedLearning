import tensorflow as tf
import numpy as np
import math

""" Rough implementation of https://arxiv.org/abs/2112.02918 """

def modify_filter(model, layer, i, j):
    kernel = model.trainable_variables[layer]
    _kernel = kernel.numpy()
    p1 = math.floor(_kernel.shape[0]/2)
    p2 = math.floor(_kernel.shape[1]/2)
    _kernel[:,:,:,i][p1,p2,j] = 1
    kernel.assign(_kernel)

def w_fully_adv_init(W, mean, std, s):
    n, L = W.shape
    
    r = np.arange(L)
    _W = W.numpy().copy()
    
    for i in range(n):
        mask = np.zeros(L)
        ids = np.random.choice(r, size=L//2, replace=False)
        mask[ids] = 1
        mask = mask.astype(bool)

        N = r[mask]
        P = r[~mask]

        zn = np.random.normal(mean, std, size=(L//2))
        zp = (-s) * zn
        np.random.shuffle(zp)
        _W[i, N] = zn
        _W[i, P] = zp

    W.assign(_W)
    
def invert_fully_g(gw, gb, i=None, epsilon=0.00001):
    b = 1. / (gb.numpy()[np.newaxis,:] + epsilon)
    w = gw.numpy().T

    if not i is None:
        x = b[:, i] * w[i, :]
    else:
        x = (np.matmul(b, w))
        print(b.shape, w.shape, x.shape)
    return x

def normalize_img(x):
    x += x.min()
    x -= x.min()
    x /= x.max()
    return x