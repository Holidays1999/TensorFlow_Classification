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
"Amp Trainer"

import time
import tqdm

import tensorflow as tf

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]


@tf.function(experimental_compile=True)
def train_one_step(model, images, target, criterion, training, optimizer):
    with tf.GradientTape() as tape:
        output = model(images, training=training)
        loss = criterion(target, output)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grad = tape.gradient(scaled_loss, model.trainable_weights)
    grads = optimizer.get_unscaled_gradients(scaled_grad)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return output, loss


@tf.function(experimental_compile=True)
def val_one_step(model, images, target, criterion, training):
    output = model(images, training=training)
    loss = criterion(target, output)

    return output, loss


def train(train_data, model, criterion, optimizer, epoch, batch_size, print_freq, writer):
    # switch to train mode
    training = True
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_data),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # Set Loss Scale
    num_batches = len(train_data)
    end = time.time()

    for i, (images, target) in tqdm.tqdm(
        enumerate(train_data), ascii=True, total=len(train_data)
    ):
        # measure data loading time
        data_time.update(time.time() - end)
        output, loss = train_one_step(model, images, target, criterion, training, optimizer)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss, images.shape[0])
        top1.update(acc1, images.shape[0])
        top5.update(acc5, images.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_data, model, criterion, print_freq, writer, epoch):

    training = False
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_data), [batch_time, losses, top1, top5], prefix="Test: "
    )

    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(val_data), ascii=True, total=len(val_data)
    ):
        # compute output

        output, loss = val_one_step(model, images, target, criterion, training)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss, images.shape[0])
        top1.update(acc1, images.shape[0])
        top5.update(acc5, images.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    progress.display(len(val_data))

    if writer is not None:
        progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg


def modifier(args, epoch, model):
    return