import numpy as np
import cv2
import matplotlib
from config import cfg
import tensorflow as tf
import random

#dataset iterator
class Dataset(object):
    def __init__(self,num_classes, dataset_type):
        self.batchsize = cfg.TRAIN.BATCHSIZE
        self.num_classes = num_classes
        self.dataaug = cfg.TRAIN.DATAAUG if dataset_type == 'train' else cfg.TEST.DATAAUG
        self.anno_path = cfg.TRAIN.ANNO_PATH if dataset_type == 'train' else cfg.TEST.ANNO_PATH
        self.inputsize = cfg.TRAIN.INPUTSIZE if dataset_type == 'train' else cfg.TEST.INPUTSIZE
        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batchsize))
        self.batch_count = 0




    def load_annotations(self):
        with open(self.anno_path, 'r') as f:
            txt = f.readline()
            annotations = [line.strip() for line in txt]
        np.random.shuffle(annotations)
        return annotations



    def parse_annotations(self, annotation):
        line = annotation.split(' ')
        img_path = line[0]
        label = int(line[1])
        image = np.array(cv2.imread(img_path))
        image = cv2.resize(image, (self.inputsize, self.inputsize), interpolation=0)
        if self.dataaug:
            image = self.random_flip(image)
        return image, label

    def __iter__(self):
        return self


    def next(self):
        with tf.device('/cpu:0'):
            batch_image = np.zeros((self.batchsize, self.inputsize, self.inputsize, 3), dtype=np.float32)
            batch_label = np.zeros((self.batchsize, self.num_classes), dtype=np.float32)
            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batchsize:
                    index = self.batch_count * self.batchsize + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, label = self.parse_annotations(annotation)
                    batch_image[num, :, :, :] = image
                    batch_label[num, label-1] = 1.0
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_flip(self, image):
        random_num = random.randint(1, 4)
        return cv2.flip(image, random_num)


    def __len__(self):
        return self.num_batches
