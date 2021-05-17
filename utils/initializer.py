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
"initializer.py for kernel initialization"

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers

from args import args


class KaimingNormal(initializers.Initializer):

    def __init__(self, mode, activation):
        self.mode = mode
        self.activation = activation

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        gain = self.calculate_gain(self.activation)
        fan = self._calculate_correct_fan(shape, self.mode)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std

        return tf.random.uniform(shape, -bound, bound, dtype=dtype)

    def calculate_gain(self, activation, param=None):
        r"""Return the recommended gain value for the given nonlinearity function.
        The values are as follows:

        ================= ====================================================
        nonlinearity      gain
        ================= ====================================================
        Linear / Identity :math:`1`
        Conv{1,2,3}D      :math:`1`
        Sigmoid           :math:`1`
        Tanh              :math:`\frac{5}{3}`
        ReLU              :math:`\sqrt{2}`
        Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
        ================= ====================================================

        Args:
            nonlinearity: the non-linear function (`nn.functional` name)
            param: optional parameter for the non-linear function

     """

        if activation.lower() == 'sigmoid':
            return 1
        elif activation.lower()== 'tanh':
            return 5.0 / 3
        if activation.lower() == 'relu':
            return math.sqrt(2.0)
        elif activation.lower()== 'leaky_relu':
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
                # True/False are instances of int, hence check above
                negative_slope = param
            else:
                raise ValueError("negative_slope {} not a valid number".format(param))
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        else:
            raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    def _calculate_correct_fan(self, shape, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

        fan_in, fan_out = self._calculate_fan_in_and_fan_out(shape)
        return fan_in if mode == 'fan_in' else fan_out

    def _calculate_fan_in_and_fan_out(self, shape):
        dimensions = len(shape)
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = shape[-2]
            fan_out = shape[-1]
        else:
            # TensorFlow Stores kernels
            # (kernel_size, kernel_size, in_channels, out_hannels)

            num_input_fmaps = shape[-2]
            num_output_fmaps = shape[-1]
            receptive_field_size = 1
            if dimensions > 2:
                receptive_field_size = np.prod(shape[:-2])
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def get_config(self):  # To support serialization
        return {"mode": self.mode, "activation": self.activation}


kaiming_normal = KaimingNormal(mode=args.mode, activation=args.activation)
HeNormal = initializers.HeNormal()
HeUniform = initializers.HeUniform()
LecunNormal = initializers.LecunNormal()
LecunUniform = initializers.LecunUniform()
Orthogonal = initializers.Orthogonal()
RandomNormal = initializers.RandomNormal()
TruncatedNormal = initializers.TruncatedNormal()