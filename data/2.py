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





    def get_loader(self, dir, train=False):
        # map classes and its idx
        self.train_classes, self.train_class_to_idx = self._find_classes(self.train_dir)
        self.val_classes, self.val_class_to_idx = self._find_classes(self.val_dir)

        # get path and labels
        self.train_data, self.tra

    def get_data_label(self, classes, class_to_idx):


    def preprocess(self, data, label):




    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def read_image(self, img_path):
        """
        Given a path of image it reads its content
        into a tf tensor
        Args:
            img_path, a tf string tensor representing the path of the image
        Returns:
            the tf image tensor
        """
        img_file = tf.io.read_file(img_path)
        return tf.image.decode_jpeg(img_file, channels=3)