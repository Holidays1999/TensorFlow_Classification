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

from tensorflow.keras import layers, Model, Sequential

from utils.builder import get_builder
from args import args


__all__ = [
    'ResNet18',
    'ResNet50',
    'ResNet101',
    'WideResNet50_2',
    'WideResNet101_2'
]


class BasicBlock(Model):
    M = 2
    expansion = 1

    def __init__(self, builder, planes, strides=1, downsample=lambda x:x, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")
        self.strides = strides
        self.conv1 = builder.conv3x3(planes, strides)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.conv2 = builder.conv3x3(planes)
        self.bn2 = builder.batchnorm()
        self.act2 = builder.activation()
        self.downsample = downsample

    def call(self, x, training=None):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out, training)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out, training)
        out += residual
        out = self.act2(out)

        return out


class Bottleneck(Model):
    M = 3
    expansion = 4

    def __init__(self, builder, planes, strides=1, downsample=lambda x:x, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(width)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.conv2 = builder.conv3x3(width, strides=strides)
        self.bn2 = builder.batchnorm()
        self.act2 = builder.activation()
        self.conv3 = builder.conv1x1(planes * self.expansion)
        self.bn3 = builder.batchnorm()
        self.act3 = builder.activation()
        self.downsample = downsample
        self.strides = strides

    def call(self, x, training):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out, training)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out, training)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out, training)

        out += residual
        out = self.act3(out)

        return out


class ResNet(Model):
    def __init__(self, builder, block, num_blocks, base_width=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.base_width = base_width

        print(f"==> Using {self.base_width // 64}x wide model")

        self.conv1 = builder.conv7x7(self.inplanes, strides=2)
        self.bn1 = builder.batchnorm()
        self.act1 = builder.activation()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(builder, block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(builder, block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(builder, block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(builder, block, 512, num_blocks[3], strides=2)
        self.avgpool = layers.GlobalAvgPool2D()
        self.classifier = layers.Dense(args.num_classes)

    def _make_layer(self, builder, block, planes, blocks, strides=1):
        downsample = lambda x : x

        if strides != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                planes * block.expansion, strides=strides
            )
            dbn = builder.batchnorm()
            downsample = Sequential([dconv, dbn])

        layer = []
        layer.append(block(builder, planes, strides, downsample, base_width=self.base_width))
        for i in range(1, blocks):
            layer.append(block(builder, planes, base_width=self.base_width))

        return Sequential(layer)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x, training)
        x = self.layer2(x, training)
        x = self.layer3(x, training)
        x = self.layer4(x, training)

        x = self.avgpool(x)
        x = self.classifier(x)

        return x


def ResNet18(**kwargs):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet50(**kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], **kwargs)


def WideResNet50_2(**kwargs):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 6, 3], base_width=64 * 2, **kwargs
    )


def WideResNet101_2(**kwargs):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 23, 3], base_width=64 * 2, **kwargs
    )