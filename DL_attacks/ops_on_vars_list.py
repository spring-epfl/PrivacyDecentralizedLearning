import tensorflow as tf
import numpy as np

""" Utility functions to perfom operations on lists/dict of tensors (e.g., variables/gradients) """

def flat_tensor_list(par):
    f = []
    for l in par:
        f.append(l.numpy().reshape(-1))
    return np.concatenate(f)

def deepCopyModel(model):
    _model = tf.keras.models.clone_model(model)
    n = len(model.variables)
    for i in range(n):
        _model.variables[i].assign(model.variables[i])        
    return _model

def clone_list_tensors(A):
    n = len(A)
    B = [None] * n
    for i in range(n):
        B[i] = tf.identity(A[i])
    return B

def assign_list_variables(A, B):
    """ A <- B """
    assert len(A) == len(B)
    n = len(A)
    for i in range(n):
        A[i].assign(B[i])

def init_list_variables(A):
    n = len(A)
    B = [None] * n
    for i in range(n):
        B[i] = tf.zeros(A[i].shape, dtype=A[i].dtype)
    return B

def agg_sum(A, B):
    assert(len(A) == len(B))
    n = len(A) 
    C = [None] * n

    for i in range(n):
        C[i] = A[i] + B[i]
        
    return C

def agg_sub(A, B):
    assert(len(A) == len(B))
    n = len(A) 
    C = [None] * n

    for i in range(n):
        C[i] = A[i] - B[i]
        
    return C

def agg_div(A, alpha):
    n = len(A) 
    C = [None] * n
    for i in range(n):
        C[i] = A[i] / alpha
    return C

def agg_neg(A):
    n = len(A) 
    C = [None] * n
    for i in range(n):
        C[i] = -A[i]
    return C

def agg_sumc(A, B):
    n = len(A) 
    C = [None] * n
    for i in range(n):
        C[i] = A[i] + B[i]
    return C

    
def select_nn_mus(keys, buff):
    """ get only a subset of a dictonary """
    new_buff = {}
    for key in keys:
        name = key.name
        new_buff[name] = buff[name]
    return new_buff