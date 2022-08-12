import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from .utils import *


class setps_lrs(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Learning-rate scheduler with fixed steps """

    def __init__(self, steps, initial_learning_rate=.1, scaling_factor=.1):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.steps = steps
        self.scaling_factor = scaling_factor

    def __call__(self, step):
        if step in self.steps:
            self.learning_rate = self.learning_rate * self.scaling_factor
        return self.learning_rate


def binary_accuracy(label, p):
    predicted = tf.argmax(p, 1, output_type=tf.int32)
    correct_prediction = tf.equal(label, predicted)
    return tf.cast(correct_prediction, tf.float32)

def cnn(input_shape, output_shape, step_slr, epsilon=0.00001):
    xin = tf.keras.layers.Input(input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(xin)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(xin, x)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    opt = lambda: tf.keras.optimizers.SGD(setps_lrs(lr, step_slr), momentum=0.9)

    eval_metrics = ['accuracy']
    loss_up_bound = -np.log(epsilon)
    
    return model, loss, opt, eval_metrics, 0


def plain_res_block(x_in, filters, stride, bn):
    
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x_in)
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(x)
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    if stride > 1:
        x_in = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(x_in)
        
        if bn:
            x_in = tf.keras.layers.BatchNormalization()(x_in)
        
    x = x + x_in
    return x

def resnet20(input_shape, output_shape, init_lr, step_slr, bn=True):
    x_in = layers.Input(input_shape)
    
    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(x_in)
   
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = plain_res_block(x, 16, 1, bn)
    x = plain_res_block(x, 16, 1, bn)
    x = plain_res_block(x, 16, 1, bn)
    
    x = plain_res_block(x, 32, 2, bn)
    x = plain_res_block(x, 32, 1, bn)
    x = plain_res_block(x, 32, 1, bn)
    
    x = plain_res_block(x, 64, 2, bn)
    x = plain_res_block(x, 64, 1, bn)
    x = plain_res_block(x, 64, 1, bn)
    
    x = layers.AveragePooling2D(pool_size=8)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(output_shape, activation='softmax')(x)
    
    model = tf.keras.Model(x_in, x)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    opt = lambda: tf.keras.optimizers.SGD(setps_lrs(step_slr, init_lr), momentum=0.9)
    
    return model, loss, opt, binary_accuracy
