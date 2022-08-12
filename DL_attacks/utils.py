import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import functools
import os, sys, glob
from subprocess import Popen, PIPE
import multiprocessing 
from multiprocessing import Pool
import functools
import time

from .ops_on_vars_list import *

def load_dataset_classification(key, split='train', buffer_size=10000):
    ds, info = tfds.load(
        key,
        split=split,
        shuffle_files=True,
        with_info=True,
        download=True
    )
    size = info.splits[split].num_examples
    try:
        num_class = info.features['label'].num_classes
    except:
        num_class = -1
    x_shape = info.features['image'].shape 
    ds = ds.shuffle(buffer_size)
    return ds, x_shape, num_class, size


def make_uniform_dataset_users(data, num_users, local_dataset_size, parse_fn, buffer_size=10000):
    
    if not parse_fn is None:
        data = data.map(parse_fn)
        
    data = data.batch(local_dataset_size)
        
    ddata = [None] * num_users
    for i, (x, y) in enumerate(data.take(num_users)):
        x = tf.data.Dataset.from_tensor_slices(x)
        y = tf.data.Dataset.from_tensor_slices(y)
        ddata[i] = tf.data.Dataset.zip((x, y))
        ddata[i] = ddata[i].shuffle(buffer_size)
   
    return ddata

def setup_data(
    load_dataset_fn,
    num_users,
    size_local_ds,
    batch_size,
    size_testset,
    type_partition,
):
    
    train, val, x_shape, num_class, parse_fn = load_dataset_fn()
    
    # create global test set
    test_set = make_uniform_dataset_users(val, 1, size_testset, parse_fn)[0].batch(batch_size)  
    
    if type_partition == 0:
         # create local datasets
        train_sets = [ds.batch(batch_size) for ds in make_uniform_dataset_users(train, num_users, size_local_ds, parse_fn)]
        size_trainset = size_local_ds * num_users
    else:
        raise Exception()
        
    return train_sets, test_set, x_shape, num_class

# ------

def setup_model(
    make_model,
    model_hparams,
    same_init,
):
    if same_init:
        f, *outs = make_model(*model_hparams)
        make_model = lambda : (deepCopyModel(f), *outs)
        return make_model
    else:
        return functools.partial(make_model, *model_hparams)

# ------
    
class EarlyStopping:
    def __init__(self, patience):
        self.best = [None, -np.inf]
        self.patience = patience
        self.current_patience = patience

    def __call__(self, i, new):
        
        if new > self.best[1]:
            self.best = i, new
            self.current_patience = self.patience
            return False
        
        if new <= self.best[1]:
            self.current_patience  -= 1
            print(f"\t {i}--getting worse {self.best[1]} --> {new} patience->{self.current_patience}")
            if self.current_patience  == 0:
                print(f"{i}--Early stop")
                return True
        return False
    
# ------

ENVGPU = "CUDA_VISIBLE_DEVICES"

def run(cmd_str, encoding='utf-8'):
    p = Popen([cmd_str], stdout=PIPE, stderr=PIPE, shell=True)
    out = p.communicate()
    stdout = out[0].decode(encoding).split('\n')
    stderr = out[1].decode()
    return stdout, stderr


def _wrapper(inputs, cmd, nGPU, GPUidShift):
    name = multiprocessing.current_process().name
    i = GPUidShift + (int(name.split('-')[-1]) -1)
    my_env = os.environ
    my_env[ENVGPU] = str(i)
    print(ENVGPU, i)
    cmd = cmd % inputs
    print('(%d)' % i, cmd)
    o, err = run(cmd)
    print(i, 'STOPPED')
    if err:
        print('\n\n', err, '\n\n')
    return o

def runMultiGPU(cmd, TASKS, nGPU=16, GPUidShift=0):
    f = functools.partial(_wrapper, cmd=cmd, nGPU=nGPU, GPUidShift=GPUidShift)
    print('nGPU', nGPU)
    with Pool(nGPU) as pool:
         out = pool.map(f, TASKS)
    return out
