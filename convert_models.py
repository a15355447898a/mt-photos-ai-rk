import os
import numpy as np
from rknn.api import RKNN

# --- 配置参数 ---

ONNX_DIR = './onnx/utils'
RKNN_DIR = './rknn/utils'
TARGET_PLATFORM = 'rk3588'



# 图像模型配置

IMG_MODEL_NAME = 'vit-b-16.img.fp32'
IMG_ONNX_PATH = os.path.join(ONNX_DIR, f'{IMG_MODEL_NAME}.onnx')
IMG_RKNN_PATH = os.path.join(RKNN_DIR, f'{IMG_MODEL_NAME}.rknn')
# 注意：rknn-toolkit2的mean/std需要乘以255

IMG_MEAN = [[0.48145466 * 255.0, 0.4578275 * 255.0, 0.40821073 * 255.0]]
IMG_STD = [[0.26862954 * 255.0, 0.26130258 * 255.0, 0.27577711 * 255.0]]



# 文本模型配置

TXT_MODEL_NAME = 'vit-b-16.txt.fp32'
TXT_ONNX_PATH = os.path.join(ONNX_DIR, f'{TXT_MODEL_NAME}.onnx')
TXT_RKNN_PATH = os.path.join(RKNN_DIR, f'{TXT_MODEL_NAME}.rknn')


def convert_image_model():

    """

    转换图像模型: ONNX -> RKNN (FP16)

    """

    print(f"--- 开始转换图像模型 (FP16): {IMG_ONNX_PATH} ---")



    if not os.path.exists(IMG_ONNX_PATH):

        print(f"[错误] ONNX 模型文件不存在: {IMG_ONNX_PATH}")

        return



    # 1. 初始化 RKNN 对象

    rknn = RKNN(verbose=True)



    # 2. 配置模型

    rknn.config(

        mean_values=IMG_MEAN,

        std_values=IMG_STD,

        target_platform=TARGET_PLATFORM,

        optimization_level=3

    )



    # 3. 加载 ONNX 模型

    print('--> 加载 ONNX 模型...')

    ret = rknn.load_onnx(model=IMG_ONNX_PATH)

    if ret != 0:

        print('加载 ONNX 模型失败!')

        rknn.release()

        return



    # 4. 构建模型 (不进行量化)

    print('--> 构建模型...')

    ret = rknn.build(do_quantization=False)

    if ret != 0:

        print('构建模型失败!')

        rknn.release()

        return



    # 5. 导出 RKNN 模型

    print(f'--> 导出 RKNN 模型到: {IMG_RKNN_PATH}')

    if not os.path.exists(RKNN_DIR):

        os.makedirs(RKNN_DIR)

    ret = rknn.export_rknn(IMG_RKNN_PATH)

    if ret != 0:

        print('导出 RKNN 模型失败!')

    else:

        print('--- 图像模型转换成功 ---')



    # 6. 释放资源

    rknn.release()

def convert_text_model():
    """
    转换文本模型: ONNX -> RKNN (FP16)
    """
    print(f"--- 开始转换文本模型 (FP16): {TXT_ONNX_PATH} ---")

    if not os.path.exists(TXT_ONNX_PATH):
        print(f"[错误] ONNX 模型文件不存在: {TXT_ONNX_PATH}")
        return

    # 1. 初始化 RKNN 对象
    rknn = RKNN(verbose=True)

    # 2. 配置模型 (文本模型不需要 mean/std)
    rknn.config(
        target_platform=TARGET_PLATFORM,
        optimization_level=3
    )

    # 3. 加载 ONNX 模型
    print('--> 加载 ONNX 模型...')
    # 文本模型的输入是 int64，工具包会自动从模型文件读取
    ret = rknn.load_onnx(model=TXT_ONNX_PATH)
    if ret != 0:
        # if 语句块内的代码需要缩进
        print('加载 ONNX 模型失败!')
        rknn.release()
        return

    # 4. 构建模型 (不进行量化)
    print('--> 构建模型...')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('构建模型失败!')
        rknn.release()
        return

    # 5. 导出 RKNN 模型
    print(f'--> 导出 RKNN 模型到: {TXT_RKNN_PATH}')
    if not os.path.exists(RKNN_DIR):
        os.makedirs(RKNN_DIR)
    ret = rknn.export_rknn(TXT_RKNN_PATH)
    if ret != 0:
        print('导出 RKNN 模型失败!')
    else:
        print('--- 文本模型转换成功 ---')

    # 6. 释放资源
    rknn.release()


if __name__ == '__main__':

    # 按照 TODO.md 的计划，创建 rknn 目录结构

    if not os.path.exists(RKNN_DIR):

        os.makedirs(RKNN_DIR)

        print(f"已创建目录: {RKNN_DIR}")



    # 运行转换

    convert_image_model()

    print("\n" + "="*50 + "\n")

    convert_text_model()
