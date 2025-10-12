# Base image for ARM64 architecture, Python 3.8
FROM --platform=linux/arm64 arm64v8/python:3.8-slim

# Set a non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- rknn_toolkit_lite2 Installation ---
# Copy the rknn_toolkit_lite2 wheel from the project structure into the image
COPY rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/

# Install the wheel file and then remove it
RUN pip install /tmp/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
    rm /tmp/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# --- Application Setup ---
# Copy application source code and requirements file
COPY rknn/ .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the application will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8060"]