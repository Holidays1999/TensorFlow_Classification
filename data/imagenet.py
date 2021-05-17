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

import tensorflow as tf
import tensorflow_datasets as tfds

class ImageNet(object):

    def __init__(self, args):
        self.data_root = args.data

        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        # Data loading Path
        self.train_dir = os.path.join(self.data_root, "train")
        self.val_dir = os.path.join(self.data_root, "val")

        # use tfds.ImageFolder to create data
        self.train_loader = tfds.ImageFolder(self.train_dir)
        self.val_loader = tfds.ImageFolder(self.val_dir)