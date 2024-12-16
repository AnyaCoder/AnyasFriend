# Stage 1: Build torch and other dependencies
FROM python:3.12-slim-bookworm AS base
WORKDIR /opt

# Install PyTorch and other dependencies
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 \
    && rm -rf /root/.cache/pip

# Build arguments
ARG TARGETARCH
ARG DEPENDENCIES="ca-certificates libsox-dev build-essential cmake libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg"

# Install system dependencies and clear apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ex \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get -y install --no-install-recommends ${DEPENDENCIES} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  

# Stage 2: anyasfriend application
FROM base AS anyasfriend
WORKDIR /opt/anyasfriend

# Copy your application files into the container
COPY . .

# Install Python dependencies and clear cache
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    set -ex \
    && pip install -e . \
    && pip install -r requirements.txt \
    && pip cache purge  

# Expose the WebSocket port
EXPOSE 8765

# Download ASR model
RUN python -m anyasfriend.components.asr.fun_asr

# Set TEMP environment variable
ENV TEMP=/tmp

# Run the main application, assuming it starts the WebSocket server
CMD ["python", "main.py"]
