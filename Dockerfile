FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

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

# Install FastAPI and uvicorn for OpenAI-compatible API server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic

# Copy handler
COPY handler.py /app/handler.py

# Expose port
ENV PORT=8008
ENV PORT_HEALTH=8008
EXPOSE 8008

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# Run the OpenAI-compatible server
CMD ["python", "/app/handler.py"]