# Copyright 2021 Zhejiang University of Techonology.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch


class CIFAR10:

    def __init__(self, args):
        super(CIFAR10, self).__init__()
        self.data_root = os.path.join(args.data, "cifar10")
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.workers = args.workers

        self.mean = tf.constant([0.491, 0.482, 0.447])
        self.std = tf.constant([0.247, 0.243, 0.262])

        self.paddings = tf.constant([[4, 4], [4, 4], [0, 0]])

        self.x_train, self.y_train = self.load_train(self.data_root)
        self.x_test, self.y_test = self.load_test(self.data_root)

        self.train_loader = self.get_loader(self.x_train, self.y_train, training=True)
        self.val_loader = self.get_loader(self.x_test, self.y_test, training=False)


    def get_loader(self, x, y, training=False):
        loader = tf.data.Dataset.from_tensor_slices((x,y))
        if training:
            loader = loader.map(self.train_preprocess, num_parallel_calls=self.workers)
            loader = loader.shuffle(len(y)).batch(self.batch_size)
        else:
            loader = loader.map(self.val_preprocess,num_parallel_calls=self.workers).batch(self.batch_size)

        return loader.prefetch(tf.data.experimental.AUTOTUNE)

    def train_preprocess(self, x, y):
        x = tf.pad(x, self.paddings)
        x = tf.image.random_crop(x, [32,32,3])
        x = tf.image.random_flip_left_right(x)
        x = tf.cast(x, dtype=tf.float32) / 255.
        x = (x - self.mean) / self.std

        y = tf.one_hot(y, depth=self.num_classes)

        return x, y

    def val_preprocess(self, x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        x = (x - self.mean) / self.std

        y = tf.one_hot(y, depth=self.num_classes)

        return x, y

    def load_train(self, data_root):
        num_train_samples = 50000
        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')
        train_dir = os.path.join(data_root, 'cifar-10-batches-py')
        for i in range(1, 6):
            fpath = os.path.join(train_dir, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
        y_train = np.array(y_train).reshape(-1).astype(np.int32)

        return x_train, y_train

    def load_test(self, data_root):
        test_dir = os.path.join(data_root, 'cifar-10-batches-py')
        x_test, y_test = load_batch(os.path.join(test_dir, 'test_batch'))
        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(0, 2, 3, 1)
        y_test = np.array(y_test).reshape(-1).astype(np.int32)

        return x_test, y_test