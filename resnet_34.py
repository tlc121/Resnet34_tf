import tensorflow as tf
import numpy as np
from common import convolutional, residual_block, Max_Pooing, fc_layer


#strucrue of res34
class Resnet34(object):
    def __init__(self, input_data, labels, trainable, classes):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.input_labels = labels


    def build_network(self):
        input = convolutional(self.input_data, filter_shape=(7,7,3,64), trainable=self.trainable,
                              downsample=True, bn=False)

        input = Max_Pooing(input, name='MaxPooling1')

        input = residual_block(input=input, input_channel=64, filter_num1=64, filter_num2=64,
                               trainable=self.trainable, name='Block1')

        input = residual_block(input=input, input_channel=64, filter_num1=64, filter_num2=64,
                               trainable=self.trainable, name='Block2')

        input = residual_block(input=input, input_channel=64, filter_num1=64, filter_num2=64,
                               trainable=self.trainable, name='Block3')

        input = residual_block(input=input, input_channel=64, filter_num1=128, filter_num2=128,
                               trainable=self.trainable, name='Block4', downsample=True)

        input = residual_block(input=input, input_channel=128, filter_num1=128, filter_num2=128,
                               trainable=self.trainable, name='Block5')

        input = residual_block(input=input, input_channel=128, filter_num1=128, filter_num2=128,
                               trainable=self.trainable, name='Block6')

        input = residual_block(input=input, input_channel=128, filter_num1=128, filter_num2=128,
                               trainable=self.trainable, name='Block7')

        input = residual_block(input=input, input_channel=128, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block8', downsample=True)

        input = residual_block(input=input, input_channel=256, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block9')

        input = residual_block(input=input, input_channel=256, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block10')

        input = residual_block(input=input, input_channel=256, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block11')

        input = residual_block(input=input, input_channel=256, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block12')

        input = residual_block(input=input, input_channel=256, filter_num1=256, filter_num2=256,
                               trainable=self.trainable, name='Block13')

        input = residual_block(input=input, input_channel=256, filter_num1=512, filter_num2=512,
                               trainable=self.trainable, name='Block14', downsample=True)

        input = residual_block(input=input, input_channel=512, filter_num1=512, filter_num2=512,
                               trainable=self.trainable, name='Block15')

        input = residual_block(input=input, input_channel=512, filter_num1=512, filter_num2=512,
                               trainable=self.trainable, name='Block16')

        avg_pool = tf.nn.avg_pool(input, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME', name='avg_pool')

        fc_in = tf.reshape(avg_pool, [-1, 512])

        prob = fc_layer(fc_in, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes)

        return prob


    def compute_loss(self):
        preds = self.build_network()
        labels = self.input_labels
        loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(preds) * labels, reduction_indices=[1]))
        correct = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accurate = tf.reduce_mean(tf.cast(correct, tf.float32))
        return loss_val, accurate


















