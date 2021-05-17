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
import pathlib
import numpy as np


def save_checkpoint(model, is_best, filename="checkpoint.h5", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    model.save_weights(filename)

    if is_best:
        model.save_weights(filename.parent / "model_best.h5")
        if not save:
            os.remove(filename)


def freeze_model_weights(model):
    """
        Although model weights are freezed, bn.moving_mean and bn.move_variance
    will still be updated
    """
    print("=> Freezing model weights")

    for layer in model.layers:
        layer.trainable = False
        print(f'==> Trainable to {layer.name} : {layer.trainable}')


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for layer in model.layers:
        layer.trainable = False
        print(f'==> Trainable to {layer.name} : {layer.trainable}')


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def get_model_size(model):
    para_num = sum([np.prod(var.shape) for var in model.trainable_variables])
    print(
        f"=> Rough estimate model params {para_num}"
    )
    return round(para_num*10**-6, 2)


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc