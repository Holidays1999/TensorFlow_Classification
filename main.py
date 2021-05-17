# Copyright 2021 Zhejiang University of Techonology
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
"main.py"

import os
import time
import copy
import random
import pathlib
import importlib
from ruamel import yaml

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.mixed_precision as mixed_precision

import data
import models
from args import args
from utils.optimizer_type import WeightDecaySGD, LossScaleOptimizerLR
from utils.schedulers import get_policy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_model_size
    )


def main():
    print(args)
    set_gpu(args)
    if args.seed is not None:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # 如果设备未在 `tf.distribute.MirroredStrategy` 的指定列表中，它会被自动检测到。
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model = get_model(args)

    # get data
    data = get_dataset(args)

    # get optimizer
    steps_per_epoch = len(list(data.train_loader))
    learning_scheduler = get_policy(args.lr_policy)(args, steps_per_epoch)
    optimizer = get_optimizer(args, model, learning_scheduler)
    if args.trainer == 'amp':
        optimizer = LossScaleOptimizerLR(optimizer, loss_scale='dynamic')

    # define loss function
    criterion = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=args.label_smoothing
    )
    print("Use label_smoothing: {} for training".format(args.label_smoothing))

    # build model
    model.build(input_shape=(None, 32, 32, 3))

    # get parameters and show model
    model_parameters = get_model_size(model)
    model.summary()

    if args.pretrained:
        model = pretrained(args, model)
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0


    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    # save a yaml file to read to record parameters
    args_text = copy.copy(args.__dict__)

    del args_text['ckpt_base_dir']
    with open(run_base_dir/'args.yaml', 'w', encoding="utf-8") as f:
        yaml.dump(
            args_text,
            f,
            Dumper=yaml.RoundTripDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4
        )

    # create recorder
    writer = tf.summary.create_file_writer(logdir=str(log_base_dir))
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        model=model,
        is_best=False,
        filename=ckpt_base_dir / f"initial.h5",
        save=False,
    )

    # Start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        modifier(args, epoch, model)
        print(f'current learning rate:{optimizer._decayed_lr(tf.float32)}')
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args.batch_size, args.print_freq, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args.print_freq, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.h5'}")
                best_time = time.time() - start_time
            save_checkpoint(
                model=model,
                is_best=is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.h5",
                save=save
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        # with writer.as_default():
        #     tf.summary.scalar("test/lr", optimizer._decayed_lr(tf.float32), epoch)
        end_epoch = time.time()

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
        conv_type=args.conv_type,
        bn_type=args.bn_type,
        arch=args.arch,
        freeze_weights=args.freeze_weights,
        trainer=args.trainer,
        model_parameters=model_parameters,
        best_time=best_time,
        batch_size=args.batch_size,
        model_optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        set=args.set,
        lr=args.learning_rate,
        epochs=args.epochs,
    )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    if args.trainer == 'amp':
        policy = mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.experimental.set_policy(policy)

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args):
    print('-------------Start Setting GPU--------------')
    # get physical gpus and gpu num
    physical_gpus = tf.config.list_physical_devices('GPU')
    print(f'==> Num of Physical GPUs:{len(physical_gpus)}')
    print(f'physical_gpus: {physical_gpus}')

    # get visiable gpus
    visiable_gpu = []
    for gpu in args.multigpu:
        visiable_gpu.append(physical_gpus[gpu])
    print(f'visiable_gpu:{visiable_gpu}')

    # set memory growth
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('------------- Complete setting memory growth--------------')

    # 设置哪个GPU对设备可见，即指定用哪个GPU
    tf.config.experimental.set_visible_devices(visiable_gpu, 'GPU')

    # 获取逻辑GPU个数
    gpus = tf.config.list_logical_devices('GPU')
    print(f'==> Num of Logical GPUs:{len(gpus)}')
    print('-------------Finishing Setting GPU--------------')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        model.load_weights(args.pretrained)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    return model


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(args, model, learning_scheduler):
    for layer in model.layers:
        if layer.trainable:
            print("<DEBUG> gradient to", layer.name)

        if not layer.trainable:
            print("<DEBUG> no gradient to", layer.name)

    if args.optimizer.lower() == "sgd":
        optimizer = WeightDecaySGD(
            learning_rate=learning_scheduler,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adam":
        optimizer = optimizers.Adam(learning_rate=learning_scheduler)
    elif args.optimizer.lower() == "adadelta":
        optimizer = optimizers.Adadelta(learning_rate=learning_scheduler)
    elif args.optimizer.lower() == "adagrad":
        optimizer = optimizers.Adagrad(learning_rate=learning_scheduler)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = optimizers.Adagrad(learning_rate=learning_scheduler)
    elif args.optimizer.lower() == "adamax":
        optimizer = optimizers.Adamax(learning_rate=learning_scheduler)
    else:
        raise ValueError("optimizer must be choosen ")

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Set, "
            "Batch Size, "
            "Base Config, "
            "Name, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Best Time, "
            "Freeze Weights, "
            "Conv Type, "
            "Bn Type, "
            "Arch, "
            "Trainer, "
            "Parameters, "
            "Optimizers, "
            "WeightDecay, "
            "Learning Rate, "
            "Epochs\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (   "{now}, "
                "{set}, "
                "{batch_size}, "
                "{base_config}, "
                "{name}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{best_time:.02f}, "
                "{freeze_weights}, "
                "{conv_type}, "
                "{bn_type}, "
                "{arch}, "
                "{trainer}, "
                "{model_parameters}, "
                "{model_optimizer}, "
                "{weight_decay}, "
                "{lr}, "
                "{epochs}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()