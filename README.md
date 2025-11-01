# MT Photos AI 识别相关任务独立部署项目 (RKNN 移植版)

本项目是 [MT-Photos/mt-photos-ai](https://github.com/MT-Photos/mt-photos-ai) 的 RKNN 移植版本，旨在让相关 AI 任务能够运行在 Rockchip NPU 硬件上。

- 基于PaddleOCR实现的文本识别(OCR)接口，利用 RKNN 进行加速。
- 基于Chinese-CLIP（OpenAI CLIP模型的中文版本）实现的图片、文本提取特征接口，利用 RKNN 进行加速。

## 更新日志

> * 2025/10/30
>   * 项目现已支持三线程并行处理，能够同时调用 RK3588 芯片的三个 NPU 核心，充分利用硬件资源
>     需要在MT-Photos中将文本识别任务和CLIP识别任务的并发数设置成3
>   * 修复了之前CLIP识别结果不准确的问题，如果使用了旧版本镜像CLIP识别，需要在系统维护工具中运行 【CLIP识别】- 清空识别结果，然后重新识别所有照片

## 目录说明

- `rknn`: 存放 RKNN 版本的核心代码、Dockerfile 及模型文件。
- `rknn-toolkit-lite2`: 存放 RKNN 运行环境相关的软件包。

## 镜像说明

您可以通过 Docker 来快速部署应用。

### 打包Docker镜像

打包前注意先下载模型文件,放到对应位置

> **模型文件说明:**
>
> 从[这里](https://github.com/a15355447898a/mt-photos-ai-rk/releases/tag/0.0)下载CLIP RKNN 模型 (`vit-b-16.img.fp32.rknn`, `vit-b-16.txt.fp32.rknn`) ,放置在 `rknn/utils` 目录下
>
> OCR模型已经放置在 `rknn/models` 目录下

```bash
# 在arm机器上打包
sudo docker build -t mt-photos-ai-rknn:latest -f rknn/Dockerfile .
```

### 运行Docker容器

```yaml
services:
    mt-photos-ai-rk:
        image: a15355447898a/mt-photos-ai-rknn:latest
        container_name: mt-photos-ai-rk
        hostname: mt-photos-ai-rk
        environment:
          - API_AUTH_KEY=1234567890
        devices:
          - /dev/dri:/dev/dri
        ports:
          - 8060:8060
        restart: always
        privileged: true
        volumes:
          - /proc/device-tree/compatible:/proc/device-tree/compatible
          - /usr/lib/librknnrt.so:/usr/lib/librknnrt.so
```

> - `API_AUTH_KEY` 为 MT Photos 连接时需要填写的 `api_key`。
> - 端口 `8060` 可根据需要自行修改。

## API

### /check

检测服务是否可用，及api-key是否正确。

```bash
curl --location --request POST 'http://127.0.0.1:8060/check' \
--header 'api-key: your_api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /ocr

文字识别。

```bash
curl --location --request POST 'http://127.0.0.1:8060/ocr' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- result.texts : 识别到的文本列表
- result.scores : 为识别到的文本对应的置信度分数，1为100%
- result.boxes : 识别到的文本位置，x,y为左上角坐标，width,height为框的宽高

```json
{
  "result": {
    "texts": [
      "识别到的文本1",
      "识别到的文本2"
    ],
    "scores": [
      "0.98",
      "0.97"
    ],
    "boxes": [
      {
        "x": "4.0",
        "y": "7.0",
        "width": "283.0",
        "height": "21.0"
      },
      {
        "x": "7.0",
        "y": "34.0",
        "width": "157.0",
        "height": "23.0"
      }
    ]
  }
}
```

### /clip/img

提取图片特征向量。

```bash
curl --location --request POST 'http://127.0.0.1:8060/clip/img' \
--header 'api-key: your_api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- results : 图片的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /clip/txt

提取文本特征向量。

```bash
curl --location --request POST 'http://127.0.0.1:8060/clip/txt' \
--header "Content-Type: application/json" \
--header 'api-key: your_api_key' \
--data '{"text":"飞机"}'
```

**response:**

- results : 文字的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /restart_v2

通过重启进程来释放内存。

```bash
curl --location --request POST 'http://127.0.0.1:8060/restart_v2' \
--header 'api-key: your_api_key'
```

**response:**

请求中断,没有返回，因为服务重启了。
