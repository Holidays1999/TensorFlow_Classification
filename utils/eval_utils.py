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
"Get topk Accuracy for evaluation"
import tensorflow as tf


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.shape[0]
    target = tf.argmax(target, axis=-1)
    
    _, pred = tf.math.top_k(output, k=maxk)
    pred = tf.transpose(pred, [1, 0])
    pred = tf.cast(pred, dtype=tf.int64)
    correct = tf.equal(tf.reshape(target, [1,-1]), pred)

    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(correct[:k], dtype=tf.float32)).numpy()
        res.append(correct_k * 100.0 / batch_size)

    return res