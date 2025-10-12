import os
import sys
import numpy as np
from PIL import Image, ImageFile
from typing import Union, List
from rknnlite.api import RKNNLite

ImageFile.LOAD_TRUNCATED_IMAGES = True

current_folder = os.path.dirname(os.path.abspath(__file__))
import bert_tokenizer as bert

def join_path(folder_path, file_name):
    return os.path.join(folder_path, file_name)

# --- 修改点: 指向 .rknn 模型 ---
model_folder_path = join_path(current_folder, "utils")
img_rknn_model_path = join_path(model_folder_path, "vit-b-16.img.fp32.rknn")
txt_rknn_model_path = join_path(model_folder_path, "vit-b-16.txt.fp32.rknn")

IMG_SIZE = 224

_tokenizer = bert.FullTokenizer()
mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# 图像预处理函数保持不变
def single_image_transform(image, image_size):
    image = Image.fromarray(np.uint8(image)).resize((image_size, image_size), Image.BICUBIC)
    image = np.array(Image.fromarray(np.uint8(image)).convert('RGB'))
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    return image.astype(np.float32)

def image_processor(image_batch, image_size=224):
    transformed_batch = [single_image_transform(img, image_size) for img in image_batch]
    transformed_batch = np.array(transformed_batch, dtype=np.float32)
    transformed_batch = np.transpose(transformed_batch, (0, 3, 1, 2))
    return transformed_batch

# 文本预处理函数保持不变
def tokenize_numpy(texts: Union[str, List[str]], context_length: int = 52) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = np.array(tokens)

    return result

# --- 修改点: 实现 rknnlite 的模型加载 ---
def load_img_model(use_dml=None): # use_dml 参数不再需要，但保持函数签名兼容
    print("--> Loading RKNN Image Model")
    engine = RKNNLite()
    engine.load_rknn(img_rknn_model_path)
    ret = engine.init_runtime(core_mask=7) # For NPU_CORE_0_1_2
    if ret != 0:
        print("Init RKNN image runtime failed.")
        exit(ret)
    print("<-- RKNN Image Model Loaded")
    return engine

def process_image(img, img_model):
    inputs = image_processor([img], image_size=IMG_SIZE)
    # --- 修改点: 使用 rknnlite 推理 ---
    outputs = img_model.inference(inputs=[inputs])
    return outputs[0][0] # 返回第一个 batch 的 embedding

# --- 修改点: 实现 rknnlite 的模型加载 ---
def load_txt_model(use_dml=None): # use_dml 参数不再需要
    print("--> Loading RKNN Text Model")
    engine = RKNNLite()
    engine.load_rknn(txt_rknn_model_path)
    ret = engine.init_runtime(core_mask=7) # For NPU_CORE_0_1_2
    if ret != 0:
        print("Init RKNN text runtime failed.")
        exit(ret)
    print("<-- RKNN Text Model Loaded")
    return engine

def process_txt(txt, text_model):
    input_data = tokenize_numpy([txt], 52)
    # --- 修改点: 使用 rknnlite 推理 ---
    outputs = text_model.inference(inputs=[input_data])
    return outputs[0][0]