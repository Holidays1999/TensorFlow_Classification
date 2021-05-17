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
"WeightDecay SGD for self Define"

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.mixed_precision.experimental import LossScaleOptimizer


class WeightDecaySGD(optimizers.SGD):
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 weight_decay=0.,
                 **kwargs):
        super(WeightDecaySGD, self).__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            **kwargs
        )
        self.weight_decay = weight_decay
        if self.weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    def apply_gradients(self, grads_and_vars, name=None):
        grads_and_vars = list(grads_and_vars)
        grads = map(self.get_grad, grads_and_vars)
        vars = map(self.get_vars, grads_and_vars)
        for grad_and_var in grads_and_vars:
            grads.append(self.get_grad(grad_and_var))
            vars.append(self.get_vars(grad_and_var))

        return super(WeightDecaySGD, self).apply_gradients(zip(grads, vars), name=name)

    def get_grad(self, grad_and_var):
        grad = tf.cond(
            pred=tf.constant('gamma' in grad_and_var[1].name or 'beta' in grad_and_var[1].name),
            true_fn=lambda: grad_and_var[0],
            false_fn=lambda: grad_and_var[0] + grad_and_var[1] * self.weight_decay
        )

        return grad

    def get_vars(self, grad_and_var):
        return grad_and_var[1]


class LossScaleOptimizerLR(LossScaleOptimizer):
    """
        AsLossScaleOptimizer didn't have function _decayed_lr to show optimizer's learning_rate,
    We just add a function for it to show it's learning_rate.
    """

    def __init__(self, optimizer, loss_scale):
        super(LossScaleOptimizerLR, self).__init__(optimizer=optimizer, loss_scale=loss_scale)

    def _decayed_lr(self, var_dtype):
        return self._optimizer._decayed_lr(var_dtype=var_dtype)