FROM nvidia/cuda:12.4.0-base-ubuntu22.04

WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    procps \
    util-linux \
    python3 python3-venv python3-pip \
    curl git wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .
RUN chmod +x run.sh

CMD ["bash", "./src/run.sh"]