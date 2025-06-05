FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory & set environment variables
RUN mkdir -p /data/.cache
ENV HF_HOME=/data/.cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV MPLCONFIGDIR=/data/.cache/matplotlib
ENV TRITON_CACHE_DIR /data/.cache/triton
RUN mkdir -p /data/.cache/triton && chmod -R 777 /data/.cache/triton

# Create and set permissions for the training output directory
# Using a generic output dir name now, as train.py will specify the -grpo suffix
RUN mkdir -p /data/model_output && chmod -R 777 /data/model_output

# Set up working directory
WORKDIR /app

# Copy requirements first for Docker caching
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# *** ADDED: Download spaCy English model ***
RUN python -m spacy download en_core_web_sm

# Attempt to install flash-attn; build continues if it fails, runtime check will handle it
RUN echo "Attempting to install flash-attn..." && \
    pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation && \
    echo "flash-attn installation command executed successfully." || \
    echo "WARNING: pip install flash-attn command had an issue or was skipped by pip."

# Copy application files
COPY app.py .
COPY train.py .
COPY train_dataset.json . 

# Expose port
EXPOSE 7860

# Run training script, then the Gradio application
CMD ["sh", "-c", "python train.py && python app.py"]
