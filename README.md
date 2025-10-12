# MT Photos AI 识别相关任务独立部署项目 (RKNN 移植版)

本项目是 [MT-Photos/mt-photos-ai](https://github.com/MT-Photos/mt-photos-ai) 的 RKNN 移植版本，旨在让相关 AI 任务能够运行在 Rockchip NPU 硬件上。

- 基于PaddleOCR实现的文本识别(OCR)接口
- 基于Chinese-CLIP（OpenAI CLIP模型的中文版本）实现的图片、文本提取特征接口，利用 RKNN 进行加速。


## 目录说明

- `rknn`: 存放 RKNN 版本的核心代码、Dockerfile 及模型文件。
- `rknn-toolkit-lite2`: 存放 RKNN 运行环境相关的软件包。

## 镜像说明

您可以通过 Docker 来快速部署应用。

### 打包Docker镜像

先得去`https://github.com/a15355447898a/mt-photos-ai-rk/releases/tag/0.0`把两个模型下载到`mt-photos-ai-rk/rknn/utils/`

```bash
# 在x86机器上使用qemu构建镜像并导出
sudo docker buildx build --platform linux/arm64 -t mt-photos-ai-rk:latest -f rknn/Dockerfile . --output type=dock
er,dest=mt-photos-ai-rk.tar
```

### 运行Docker容器

```yaml
services:
    mt-photos-ai-rk:
        image: mt-photos-ai-rk:latest
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

## 下载源码本地运行

### 环境要求

- Python 3.8+
- Rockchip NPU 驱动已正确安装
- RKNN Toolkit Lite 2

### 步骤

1.  **安装 Python 依赖**
    ```bash
    cd rknn
    pip install -r requirements.txt
    ```
    同时，请根据您的设备和操作系统，安装正确的 `rknn_toolkit_lite2` 版本。例如，在 aarch64 架构的系统上，可以安装 `rknn-toolkit-lite2/packages/` 目录下的 whl 包：
    ```bash
    pip install ../rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    ```

2.  **配置环境变量**
    复制 `.env.example` 文件为 `.env`，并修改其中的 `API_AUTH_KEY`。

3.  **启动服务**
    ```bash
    python server.py
    ```

看到以下日志，则说明服务已经启动成功：
```bash
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8060 (Press CTRL+C to quit)
```

> **模型文件说明:**
>
> CLIP RKNN 模型 (`vit-b-16.img.fp32.rknn`, `vit-b-16.txt.fp32.rknn`) 已内置在 `rknn/utils` 目录下。

## API

API 接口与原项目保持一致。

### /check

检测服务是否可用，及api-key是否正确

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

**注意：当前 RKNN 版本暂未实现 OCR 功能。**

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