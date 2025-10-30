# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import cv2
import numpy as np
from .utils import operators
from .utils.db_postprocess import DBPostProcess, DetPostProcess
from .rknn_executor import RKNN_model_container


DET_INPUT_SHAPE = [480, 480] # h,w

PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': {
                    'image_shape': DET_INPUT_SHAPE
                }
         },
        {
            'NormalizeImage':
            {
                    'std': [1., 1., 1.],
                    'mean': [0., 0., 0.],
                    'scale': '1.',
                    'order': 'hwc'
            }
        }
        ]

POSTPROCESS_CONFIG = {
    'DBPostProcess':{
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
    }
}

class TextDetector:
    def __init__(self, model_path, target='rk3588', core_mask=-1) -> None:
        self.model = RKNN_model_container(model_path, target=target, core_mask=core_mask)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        return img

    def run(self, img):
        model_input = self.preprocess({'image':img})
        # Add batch dimension for RKNN inference (needs 4D input)
        input_data = np.expand_dims(model_input['image'], axis=0)
        output = self.model.run([input_data])

        preds = {'maps' : output[0].astype(np.float32)}
        result = self.db_postprocess(preds, model_input['shape'])

        output = self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)
        return output
