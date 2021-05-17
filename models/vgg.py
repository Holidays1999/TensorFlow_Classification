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
"VGG Model for Classification"

from tensorflow.keras import layers, Sequential, Model

from utils.builder import get_builder
from args import args


__all__ = [
    'vgg11_bn',
    'vgg11',
    'vgg13_bn',
    'vgg13',
    'vgg16_bn',
    'vgg16',
    'vgg19_bn',
    'vgg19'
]


class VGG(Model):
    '''
    VGG model
    '''

    def __init__(self, builder, features, drop_rate=0.5):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = Sequential([
            layers.Dropout(drop_rate),
            layers.Dense(512, use_bias=False),
            builder.activation(),
            layers.Dropout(drop_rate),
            layers.Dense(512, use_bias=False),
            builder.activation()
        ])
        self.classifier.add(layers.Dense(args.num_classes))

    def call(self, x, training=None):
        x = self.features(x, training)
        x = self.classifier(x, training)
        return x


def make_layers(cfg, builder, batch_norm=False):
    layer = []
    for index, v in enumerate(cfg):
        if v == 'M':
            layer += [layers.MaxPool2D(pool_size=2, strides=2, padding='same')]
        else:
            conv2d = builder.conv3x3(v)
            if batch_norm:
                layer += [conv2d, builder.batchnorm(), builder.activation()]
            else:
                layer += [conv2d, builder.activation()]
    layer += [layers.GlobalAvgPool2D()]
    return Sequential(layer)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(get_builder(), make_layers(cfg['A'], get_builder()))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['A'], get_builder(), batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(get_builder(), make_layers(cfg['B'], get_builder()))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['B'], get_builder(), batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(get_builder(), make_layers(cfg['D'], get_builder()))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(get_builder(),make_layers(cfg['D'], get_builder(), batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(get_builder(), make_layers(cfg['E'], get_builder()))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['E'], get_builder(), batch_norm=True))