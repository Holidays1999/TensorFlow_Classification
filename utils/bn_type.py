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
"Define BatchNorm for Models"

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization


LearnedBatchNorm = layers.BatchNormalization

class NonCenterScaleLearnedBatchNorm(BatchNormalization):
    def __init__(self, axis=-1):
        super(NonCenterScaleLearnedBatchNorm, self).__init__(axis=axis, scale=False, center=False)