FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle GPU 3.2.1 from official repo (CUDA 12.6 compatible with 12.4)
RUN pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install PaddleX (latest 3.3.11) and PaddleOCR with doc-parser
RUN pip install --no-cache-dir \
    paddlex==3.3.11 \
    "paddleocr[doc-parser]" \
    runpod

# Install special safetensors for PaddleOCR-VL
RUN pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Note: Model will be downloaded on first request (cold start)
# Cannot pre-download during build as CUDA is not available

# Copy handler
COPY handler.py /handler.py

# Run the handler
CMD ["python", "/handler.py"]
