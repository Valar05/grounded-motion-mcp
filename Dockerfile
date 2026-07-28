FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG GROUNDED_MOTION_REVISION=unknown
ARG GROUNDED_MOTION_CHECKPOINT_SHA256=f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GROUNDED_MOTION_WORKSPACE=/data \
    GROUNDED_MOTION_MODEL_CACHE=/models \
    GROUNDED_MOTION_TRANSPORT=streamable-http \
    GROUNDED_MOTION_HOST=0.0.0.0 \
    GROUNDED_MOTION_PORT=8080 \
    GROUNDED_MOTION_REVISION=${GROUNDED_MOTION_REVISION} \
    GROUNDED_MOTION_CHECKPOINT_SHA256=${GROUNDED_MOTION_CHECKPOINT_SHA256}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
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
       --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install openmim==0.3.9 \
    && mim install "mmcv==2.1.0" \
    && python3 -m pip install "mmengine==0.10.7" "mmpose==1.3.2"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install .

RUN mkdir -p /data /models \
    && curl --fail --location --retry 3 \
       --output /models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
       https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
    && printf '%s  %s\n' \
       "$GROUNDED_MOTION_CHECKPOINT_SHA256" \
       /models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
       | sha256sum --check --strict

VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["grounded-motion-mcp"]
