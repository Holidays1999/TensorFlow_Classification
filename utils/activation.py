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
"Define Activations for Models"

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


__all__ = [
    'ReLU',
    'ELU',
    'SELU'
]

ReLU = layers.ReLU
ELU = layers.ELU


class SELU(Layer):
    "SELU Activation for Model"

    def __init__(self):
        super(SELU, self).__init__()
        self.elu = layers.ELU()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def call(self, x):
        return self.scale * tf.where(x>=0, x, self.alpha*self.elu(x))