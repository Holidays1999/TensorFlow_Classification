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
"Layers Builder for Models"

from args import args

import tensorflow as tf

import utils.conv_type
import utils.bn_type
import utils.activation
import utils.initializer


class Builder(object):
    def __init__(self, conv_layer, bn_layer, activation, init):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.activation = activation
        self.init = init

    def conv(self, kernel_size, out_planes, strides=1, padding='same'):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(
                out_planes,
                kernel_size=3,
                strides=strides,
                padding=padding,
                use_bias=False,
                kernel_initializer=self.init
            )
        elif kernel_size == 1:
            conv = conv_layer(
                out_planes,
                kernel_size=1,
                strides=strides,
                padding=padding,
                use_bias=False,
                kernel_initializer=self.init
            )
        elif kernel_size == 5:
            conv = conv_layer(
                out_planes,
                kernel_size=5,
                strides=strides,
                padding=padding,
                use_bias=False,
                kernel_initializer=self.init
            )
        elif kernel_size == 7:
            conv = conv_layer(
                out_planes,
                kernel_size=7,
                strides=strides,
                padding=padding,
                use_bias=False,
                kernel_initializer=self.init
            )
        elif kernel_size == 11:
            conv = conv_layer(
                out_planes,
                kernel_size=11,
                strides=strides,
                padding=padding,
                use_bias=False,
                kernel_initializer=self.init
            )
        else:
            return None

        return conv

    def conv11x11(self, out_planes, strides=1, padding='same', **kwargs):
        """3x3 convolution with padding"""
        c = self.conv(11, out_planes, strides=strides, padding=padding,  **kwargs)
        return c

    def conv3x3(self, out_planes, strides=1, padding='same', **kwargs):
        """3x3 convolution with padding"""
        c = self.conv(3, out_planes, strides=strides, padding=padding,  **kwargs)
        return c

    def conv1x1(self, out_planes, strides=1, padding='same', **kwargs):
        """1x1 convolution with padding"""
        c = self.conv(1, out_planes, strides=strides, padding=padding, **kwargs)
        return c

    def conv7x7(self, out_planes, strides=1, padding='same', **kwargs):
        """7x7 convolution with padding"""
        c = self.conv(7, out_planes, strides=strides, padding=padding, **kwargs)
        return c

    def conv5x5(self, out_planes, strides=1, padding='same', **kwargs):
        """5x5 convolution with padding"""
        c = self.conv(5, out_planes, strides=strides, padding=padding, **kwargs)
        return c

    def batchnorm(self, **kwargs):
        """batchnormalization"""
        return self.bn_layer(**kwargs)

    def activation(self):
        return self.activation()


def get_builder():

    print("==> Conv Type: {}".format(args.conv_type))
    print("==> BN Type: {}".format(args.bn_type))
    print("==> Activation Type: {}".format(args.activation))
    print("==> Initialization Type: {}".format(args.init))

    conv_layer = getattr(utils.conv_type, args.conv_type)
    bn_layer = getattr(utils.bn_type, args.bn_type)
    activation = getattr(utils.activation, args.activation)
    init = getattr(utils.initializer, args.init)

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, activation=activation, init=init)

    return builder
