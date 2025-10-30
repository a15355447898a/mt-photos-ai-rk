from dotenv import load_dotenv
import os
import sys
import threading
import queue
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
from pydantic import BaseModel
import clip as clip
from ocr import TextSystem
from rknnlite.api import RKNNLite

# import onnxruntime as ort
# device = ort.get_device()
# print(f"Using device: {device}")

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
# env_use_dml = os.getenv("MT_USE_DML", "on") == "on" # 是否启用dml加速，当使用onnxruntime-directml加速时，使用这行
env_use_dml = False
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

ocr_models = queue.Queue()
clip_img_models = queue.Queue()
clip_txt_models = queue.Queue()
restart_timer = None


class LazyModelSlot:
    """延迟加载模型，避免启动时占用额外内存，同时保持队列可用。"""

    def __init__(self, factory, preload=False):
        self._factory = factory
        self._model = None
        self._lock = threading.Lock()
        if preload:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._factory()
        return self._model

    def get_model(self):
        return self.ensure_loaded()

# RKNN OCR model paths
DET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocrv4_det.rknn')
REC_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocrv4_rec.rknn')
CHARACTER_DICT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocr_keys_v1.txt')
RKNN_TARGET = os.getenv("RKNN_TARGET", "rk3588")

class ClipTxtRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    # Initialize and populate the model queues
    for i in range(3):
        core_mask = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][i]
        
        # OCR model
        ocr_model = TextSystem(
            det_model_path=DET_MODEL_PATH,
            rec_model_path=REC_MODEL_PATH,
            character_dict_path=CHARACTER_DICT_PATH,
            target=RKNN_TARGET,
            drop_score=0.5,
            core_mask=core_mask
        )
        ocr_models.put(ocr_model)
        
        # CLIP image model
        img_model = clip.load_img_model(use_dml=env_use_dml, core_mask=core_mask)
        clip_img_models.put(img_model)
        
        # CLIP text model（懒加载保证队列不为空）
        txt_model_slot = LazyModelSlot(
            lambda mask=core_mask: clip.load_txt_model(use_dml=env_use_dml, core_mask=mask),
            preload=env_auto_load_txt_modal
        )
        clip_txt_models.put(txt_model_slot)


@app.middleware("http")
async def check_activity(request, call_next):
    global restart_timer

    if restart_timer:
        restart_timer.cancel()

    restart_timer = threading.Timer(server_restart_time, restart_program)
    restart_timer.start()

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))


def trans_result(filter_boxes, filter_rec_res):
    texts = []
    scores = []
    boxes = []
    if filter_boxes is None or filter_rec_res is None:
        return {'texts': texts, 'scores': scores, 'boxes': boxes}

    for dt_box, rec_result in zip(filter_boxes, filter_rec_res):
        text, score = rec_result[0]
        box = {
            'x': to_fixed(dt_box[0][0]),
            'y': to_fixed(dt_box[0][1]),
            'width': to_fixed(dt_box[1][0] - dt_box[0][0]),
            'height': to_fixed(dt_box[2][1] - dt_box[0][1])
        }
        boxes.append(box)
        texts.append(text)
        scores.append(f"{score:.2f}")
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "env_use_dml":env_use_dml
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 客户端可调用，触发重启进程来释放内存，OCR过程中会触发这个请求；新版本OCR内存增长正常了，此方法不执行
    # restart_program()
    return {'result': 'pass'}

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    ocr_model = ocr_models.get()
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        # Run RKNN OCR
        filter_boxes, filter_rec_res = ocr_model.run(img)
        result = trans_result(filter_boxes, filter_rec_res)
        del img
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        ocr_models.put(ocr_model)

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    clip_img_model = clip_img_models.get()
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = await predict(clip.process_image, img, clip_img_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        clip_img_models.put(clip_img_model)

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    clip_txt_slot = clip_txt_models.get()
    try:
        text = request.text
        clip_txt_model = clip_txt_slot.get_model()
        result = await predict(clip.process_txt, text, clip_txt_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        clip_txt_models.put(clip_txt_slot)

async def predict(predict_func, inputs,model):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs,model)

def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)
