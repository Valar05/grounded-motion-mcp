FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GROUNDED_MOTION_WORKSPACE=/data \
    GROUNDED_MOTION_MODEL_CACHE=/models \
    GROUNDED_MOTION_TRANSPORT=streamable-http \
    GROUNDED_MOTION_HOST=0.0.0.0 \
    GROUNDED_MOTION_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       build-essential \
       git \
       python3 \
       python3-dev \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
       torch==2.4.1 \
       torchvision==0.19.1 \
       --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install openmim==0.3.9 \
    && mim install "mmcv==2.1.0" \
    && python3 -m pip install "mmengine==0.10.7" "mmpose==1.3.2"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install .

RUN mkdir -p /data /models
VOLUME ["/data", "/models"]
EXPOSE 8000

ENTRYPOINT ["grounded-motion-mcp"]
