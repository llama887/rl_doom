# Base image with Python and GPU support
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# Copy your requirements.txt first (this is to cache dependencies installation)
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt