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

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from args import args


__all__ = ["ConstanrLr", "CosineLr", "get_policy"]


def get_policy(name):
    if name is None:
        return ConstanrLr

    out_dict = {
        "constant_lr": ConstanrLr,
        "cosine_lr": CosineLr,
    }
    print("Use Learning Rate Polict: {} for training".format(args.gpu))
    return out_dict[name]


class ConstanrLr(LearningRateSchedule):
    "ConstanrLr"

    def __init__(self, args, step_per_epoch):
        super(ConstanrLr, self).__init__()
        self.initial_learning_rate = args.learning_rate
        self.warmup_length = args.warmup_length
        self.step_per_epoch = step_per_epoch

    def call(self, steps):
        learning_rate = tf.cond(
            pred = tf.less(steps // self.step_per_epoch, self.warmup_length),
            true_fn = lambda: self._warmup_lr(self.initial_learning_rate, self.warmup_length, steps // self.step_per_epoch),
            false_fn = lambda: tf.identity(learning_rate)
        )
        return learning_rate

    def _warmup_lr(self, base_lr, warmup_length, epoch):
        return base_lr * (epoch + 1) / warmup_length


class CosineLr(LearningRateSchedule):
    "ConstanrLr"

    def __init__(self, args, step_per_epoch):
        super(CosineLr, self).__init__()
        self.initial_learning_rate = args.learning_rate
        self.warmup_length = args.warmup_length
        self.step_per_epoch = step_per_epoch
        self.epochs = args.epochs

    def __call__(self, steps):
        learning_rate = tf.cond(
            pred=tf.less(steps // self.step_per_epoch, self.warmup_length),
            true_fn= lambda: self._warmup_lr(self.initial_learning_rate, self.warmup_length, steps // self.step_per_epoch),
            false_fn = lambda: self.lr_decay(steps)
        )

        return learning_rate

    def lr_decay(self, steps):
        e = steps // self.step_per_epoch - self.warmup_length
        es = self.epochs - args.warmup_length
        learning_rate = 0.5 * (1 + tf.math.cos(np.pi * e / es)) * self.initial_learning_rate

        return learning_rate

    def _warmup_lr(self, base_lr, warmup_length, epoch):
        return base_lr * (epoch + 1) / warmup_length