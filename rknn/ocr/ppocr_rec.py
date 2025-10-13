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
from .utils.rec_postprocess import CTCLabelDecode
from .rknn_executor import RKNN_model_container


REC_INPUT_SHAPE = [48, 320] # h,w

PRE_PROCESS_CONFIG = [
        {
            'NormalizeImage': {
                'std': [1, 1, 1],
                'mean': [0, 0, 0],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }
        ]


class TextRecognizer:
    def __init__(self, model_path, character_dict_path, target='rk3588') -> None:
        self.model = RKNN_model_container(model_path, target=target)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        POSTPROCESS_CONFIG = {
            'CTCLabelDecode':{
                "character_dict_path": character_dict_path,
                "use_space_char": True
            }
        }
        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG['CTCLabelDecode'])

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        return img

    def run(self, imgs):
        outputs=[]
        for img in imgs:
            img = cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
            model_input = self.preprocess({'image':img})
            # Add batch dimension for RKNN inference (needs 4D input)
            input_data = np.expand_dims(model_input['image'], axis=0)
            output = self.model.run([input_data])
            preds = output[0].astype(np.float32)
            output = self.ctc_postprocess(preds)
            outputs.append(output)
        return outputs
