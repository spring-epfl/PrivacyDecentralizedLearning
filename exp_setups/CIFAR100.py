from . import *
from DL_attacks.utils import load_dataset_classification
import tensorflow as tf

dsk = 'cifar100'
ds_size = 50000

# model's arch
model_maker = model.resnet20

# batch size for Distributed SGD
batch_size = 64
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // nu

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 10000
    
# function used to pre-process data
@tf.function
def parse_img(batch):
    x = batch['image']
    x = tf.cast(x, tf.float32)
    # normalize in [-1, 1]
    x = (x / (255/2) - 1)
    
    y = batch['label']
    y = tf.cast(y, tf.int32)
    return x, y

# function to load the dataset
def load_dataset():
    train, x_shape, num_class, tot_train_size = load_dataset_classification(dsk, 'train')
    val, _, _, tot_test_size = load_dataset_classification(dsk, 'test')
    return train, val, x_shape, num_class, parse_img