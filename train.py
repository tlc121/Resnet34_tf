import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from resnet_34 import Resnet34
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from config import cfg
from dataset import Dataset

class Res34_train(object):
    def __init__(self):
        self.Batch_Size = cfg.TRAIN.BATCHSIZE
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.num_classes = cfg.TRAIN.NUMCLASS
        self.train_data = Dataset(self.num_classes,'train')
        self.test_data = Dataset(self.num_classes,'test')
        self.sess = tf.Session()
        self.initial_weights = cfg.TRAIN.INITIAL_WEIGHT
        self.epoch = cfg.TRAIN.EPOCH


        #inputdata
        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.input_labels = tf.placeholder(dtype=tf.float32, name='input_labels')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')

        with tf.name_scope('defince_loss'):
            self.model = Resnet34(input_data=self.input_data, labels=self.input_labels, trainable=self.trainable, classes=self.num_classes)
            self.loss, self.accurate = self.model.compute_loss()
            self.net_var = tf.global_variables()

        with tf.name_scope('optimizer'):
            self.optimizer =  tf.train.AdamOptimizer(self.learn_rate_init).minimize(self.loss)


        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        keep_prob = tf.placeholder(dtype=tf.float32, name='dropout')
        try:
            print ('=>Restore weights from' + self.initial_weights)
            self.loader.restore(self.sess, self.initial_weights)
        except:
            print (self.initial_weights + 'does not exist!')
            print ('=>starts training from scratch ...')

        for epoch in range(self.epoch):
            pabr = tqdm(self.train_data)
            train_epoch_loss, test_epoch_loss = [], []
            train_epoch_acc, test_epoch_acc = [], []
            for train_data in pabr:
                self.sess.run(self.optimizer, feed_dict={self.input_data: train_data[0],
                                                         self.input_labels: train_data[1],
                                                         self.trainable: True,
                                                         keep_prob: cfg.TRAIN.DROPOUT})
                train_step_loss, train_step_acc = self.sess.run([self.loss, self.accurate],
                                                                feed_dict={self.input_data: train_data[0],
                                                                self.input_labels: train_data[1], self.trainable: True})

                train_epoch_loss.append(train_step_loss)
                train_epoch_acc.append(train_step_acc)
                pabr.set_description("train loss: %.2f" %train_step_loss)


            for test_data in self.test_data:
                test_step_loss, test_step_acc = self.sess.run([self.loss, self.accurate],
                                                                feed_dict={self.input_data: test_data[0],
                                                                           self.input_labels: test_data[1],
                                                                           self.trainable: False})

                test_epoch_loss.append(test_step_loss)
                test_epoch_acc.append(test_step_acc)

            train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = np.mean(train_epoch_loss), np.mean(train_epoch_acc), np.mean(test_epoch_loss), np.mean(test_epoch_acc)
            print ('Epoch: %2d Train loss: %.2f Train acc: %.2f'
                   %(epoch, train_epoch_loss, train_epoch_acc))

            print ('Test loss: %.2f Test acc: %.2f'
                   % (test_epoch_loss, test_epoch_acc))

            if epoch % 10 == 0 and epoch >= 20:
                ckpt_file = '/data/tlc/model_res34/checkpoint/res34_test_loss=%.4f.ckpt' %test_epoch_loss
                self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    Res34_train().train()



