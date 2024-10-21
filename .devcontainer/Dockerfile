# Base image with Python and GPU support
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# Install required system packages for vizdoom and OpenGL support
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libboost-all-dev \
    libopencv-dev \
    x11-xserver-utils \
    x11-apps \
    mesa-utils \
    python3-opengl

# Install GitHub CLI (gh)
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Copy your requirements.txt first (this is to cache dependencies installation)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Start Xvfb for a virtual display
RUN echo "#!/bin/bash\nXvfb :1 -screen 0 800x600x16 &\nexport DISPLAY=:1\nexec \"\$@\"" > /usr/local/bin/xvfb-run-docker && \
    chmod +x /usr/local/bin/xvfb-run-docker

# Set the entry point to always run Xvfb
ENTRYPOINT ["xvfb-run-docker"]
