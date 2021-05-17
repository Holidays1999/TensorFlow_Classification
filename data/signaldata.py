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


class SignalData:

    def __init__(self, args):
        super(SignalData, self).__init__()
        self.data_root = args.data
        self.SNR = args.SNR
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.function_norm = self.get_function_norm(args.norm)
        
        # Data loading Path
        train_dir = os.path.join(self.data_root, "train")
        val_dir = os.path.join(self.data_root, "val")


        # Get data loader
        self.train_loader = self.get_loader(train_dir, train=True)
        self.val_loader = self.get_loader(val_dir, train=False)


    def get_loader(self, dir, train):
        # make sure in tensrflow you should use channel_last config
        data, label, snr = self.load_data(dir)

        # get suitable SNR data for training and testing
        data = data[self.SNR <= snr]
        label = label[self.SNR <= snr]

        # check the data length
        assert len(data) == len(label), 'data and label must have the same length'

        # create dataset loader
        loader = tf.data.Dataset.from_tensor_slices([data, label])

        # map preprocess and batchsize
        loader = loader.map(self.preprocess).batch(self.batch_size)
        # shuffle training data
        if train:
            loader = loader.shuffle(buffer_size=self.batchsize*100)
        return loader

    def preprocess(self, data, label):
        data = self.function_norm(data)
        label = tf.one_hot(label, depth=self.num_classes)
        return data, label

    def load_data(self, dir):
        files = [os.path.join(dir, file) for file in os.listdir(dir)]
        assert len(files) == 2
        # path[0] ==> data   path[1][1] ==> label   path[1][0] ==> snr
        if 'X' not in files[0].split('/')[-1].upper():
            files = files[::-1]
        # for tensorflow we use channel last, so dims should be swap
        data = np.load(files[0]).astype(np.float32).transpose(0,2,1)
        label = np.load(files[1])[1].astype(np.float32).astype(np.int32).reshape(-1)
        snr = np.load(files[1])[0].astype(np.float32).astype(np.int32).reshape(-1)
        return data, label, snr

    def get_function_norm(self, norm):
        if norm == 'L1':
            return self.L1
        elif norm == 'L2':
            return self.L2
        else:
            raise ValueError("Invalid norm choice")

    def L1(self, data):
        return data / tf.math.reduce_max(tf.math.abs(data),axis=0)

    def L2(self, data):
        norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(data), axis=0))
        return data / norm