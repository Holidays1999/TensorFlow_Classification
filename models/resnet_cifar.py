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
"ResNet Model for Classification"

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from utils.builder import get_builder
from args import args


__all__ = [
    'cResNet18',
    'cResNet50',
    'cResNet101',
    'cResNet152',
]


class BasicBlock(Model):
    expansion = 1

    def __init__(self, builder, in_planes, planes, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(planes, strides=strides)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.conv2 = builder.conv3x3(planes, strides=1)
        self.bn2 = builder.batchnorm()
        self.act2 = builder.activation()

        self.shortcut = Sequential()
        if strides != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential([
                builder.conv1x1(self.expansion * planes, strides=strides),
                builder.batchnorm(),
            ])

    def call(self, x, training):
        out = self.act1(self.bn1(self.conv1(x),training))
        out = self.bn2(self.conv2(out),training)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(Model):
    expansion = 4

    def __init__(self, builder, in_planes, planes, strides=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(planes)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.conv2 = builder.conv3x3(planes, strides=strides)
        self.bn2 = builder.batchnorm()
        self.act2 = builder.activation()
        self.conv3 = builder.conv1x1(self.expansion * planes)
        self.bn3 = builder.batchnorm()
        self.act3 = builder.activation()

        self.shortcut = Sequential()
        if strides != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential([
                builder.conv1x1(self.expansion * planes, strides=strides),
                builder.batchnorm(),
            ])

    def call(self, x, training):
        out = self.act1(self.bn1(self.conv1(x), training))
        out = self.act2(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out += self.shortcut(x)
        out = self.act3(out)

        return out


class ResNet(Model):
    def __init__(self, builder, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder
        self.conv1 = builder.conv3x3(self.in_planes)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avgpool = layers.GlobalAvgPool2D()

        self.fc = layers.Dense(args.num_classes)

    def _make_layer(self, block, planes, num_blocks, strides):
        strides_collect = [strides] + [1] * (num_blocks - 1)
        layer = []
        for strides in strides_collect:
            layer.append(block(self.builder, self.in_planes, planes, strides))
            self.in_planes = planes * block.expansion

        return Sequential(layer)

    @tf.function
    def call(self, x, training=None):

        out = self.conv1(x)
        out = self.bn1(out, training)
        out = self.act1(out)

        out = self.layer1(out, training)
        out = self.layer2(out, training)
        out = self.layer3(out, training)
        out = self.layer4(out, training)
        out = self.avgpool(out)
        out = self.fc(out)

        return out


def cResNet18(**kwargs):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], **kwargs)


def cResNet50(**kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], **kwargs)


def cResNet101(**kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], **kwargs)


def cResNet152(**kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3], **kwargs)