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
"DenseNet Model for Classification"

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

from utils.builder import get_builder
from args import args


__all__ = [
    'DenseNet121',
    'DenseNet161',
    'DenseNet169',
    'DenseNet201',
    'densenet_BC_100',
]


class _DenseLayer(Sequential):
    def __init__(self, builder, bn_size, growth_rate):
        super(_DenseLayer, self).__init__()

        self.add(builder.batchnorm())
        self.add(builder.activation())
        self.add(builder.conv1x1(bn_size*growth_rate))
        self.add(builder.batchnorm())
        self.add(builder.activation())
        self.add(builder.conv1x1(growth_rate))

    def call(self, x, training):
        new_features = super(_DenseLayer, self).call(x, training)
        return tf.concat([x, new_features], -1)


class _DenseBlock(Sequential):
    def __init__(self, builder, num_layers, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add(_DenseLayer(builder, bn_size, growth_rate))


class _Transition(Sequential):
    def __init__(self, builder, out_channels):
        super(_Transition, self).__init__()
        self.add(builder.batchnorm())
        self.add(builder.activation())
        self.add(builder.conv1x1(out_channels))
        self.add(layers.AveragePooling2D(pool_size=2, strides=2))


class DenseNet(Model):

    def __init__(self, builder, growth_rate=12, block_config=(6,12,24,16), bn_size=4, theta=0.5):
        super(DenseNet, self).__init__()
        num_init_feature = 2 * growth_rate


        self.features = Sequential([
            builder.conv7x7(num_init_feature, strides=2),
            builder.batchnorm(),
            builder.activation(),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same')
            ])

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add(_DenseBlock(builder, num_layers, bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add(_Transition(builder, int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add(builder.batchnorm())
        self.features.add(builder.activation())
        self.features.add(layers.GlobalAvgPool2D())
        self.classifier = layers.Dense(args.num_classes)

    def call(self, x, training):
        features = self.features(x, training)
        out = self.classifier(features)
        return out


# DenseNet for ImageNet
def DenseNet121():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 24, 16))


def DenseNet169():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 32, 32))


def DenseNet201():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 48, 32))


def DenseNet161():
    return DenseNet(get_builder(), growth_rate=48, block_config=(6, 12, 36, 24))


# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet(get_builder(), growth_rate=12, block_config=(16, 16, 16))