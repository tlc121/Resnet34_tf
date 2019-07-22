import numpy as np
import tensorflow as tf

def convolutional(input, filter_shape, trainable, name, downsample=False, activation=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            strides = [1,2,2,1]
            padding = 'VALID'
        else:
            strides = [1,1,1,1]
            padding = 'SAME'

        weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=trainable,
                                  shape=filter_shape, initializer=tf.contirb.layers.xavier_initializer())

        conv = tf.nn.conv2d(input=input, filter=weights, strides=strides, padding=padding)
        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                       gamma_initializer=tf.ones_initializer(),
                                                       moving_mean_initializer=tf.zeros_initializer(),
                                                       moving_variance_initializer=tf.ones_initializer(),
                                                       trainable=True)

        else:
            bias = tf.get_variable(name='bias', shape=filter_shape[-1], trainable=trainable,
                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            conv = tf.nn.bias_add(conv, bias)

        if activation:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        return conv

def Max_Pooing(input, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=(1,3,3,1), strides=[1,2,2,1], padding='SAME', name='layer1')

def fc_layer(input, name, trainable, num_classes):
    with tf.variable_scope(name):
        fc_34 = tf.get_variable(name='weight', dtype=tf.float32, trainable=trainable,
                                shape=[512, num_classes], initializer=tf.contirb.layers.xavier_initializer())
        bias_34 = tf.get_variable(name='bias', shape=num_classes, trainable=trainable,
                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        res = tf.matmul(input, fc_34) + bias_34

        return tf.nn.softmax(res)


def residual_block(input, input_channel, filter_num1, filter_num2, trainable, name, downsample=False):
    with tf.variable_scope(name):
        if downsample:
            short_cut = convolutional(input, filter_shape=(3, 3, input_channel, filter_num1),
                                             trainable=trainable, bn=False,
                                             name = 'kernel_weight', downsample=downsample)
        else:
            short_cut = input

        input = convolutional(input, filter_shape=(3,3,input_channel,filter_num1), trainable=True, name='conv1', downsample=downsample)
        input = convolutional(input, filter_shape=(3,3,filter_num1,filter_num2), trainable=True, name='conv2')
        residual_output = input + short_cut
        return residual_output









