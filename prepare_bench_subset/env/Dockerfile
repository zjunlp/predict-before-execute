FROM johnsonzheng03/predict-before-execute

# Ensure bash is used for RUN and ENTRYPOINT context if needed
SHELL ["/bin/bash", "-lc"]

# Create app directories and HF caches
RUN mkdir -p /app/input /app/submission /app/workspaces /app/.cache/huggingface/transformers

# Hugging Face mirrors and caches (can be overridden at runtime)
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HUGGINGFACE_HUB_ENDPOINT=${HF_ENDPOINT}
ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
# Increase default request timeout/retries for hub downloads (can be overridden at runtime)
ENV HF_HUB_REQUEST_TIMEOUT=60
ENV HF_HUB_DOWNLOAD_RETRIES=5
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONUNBUFFERED=1
# Ensure Python can import local packages under /app
ENV PYTHONPATH=/app

# Optional: install git-lfs for repos needing LFS
# RUN apt-get update && apt-get install -y git-lfs && git lfs install --system && rm -rf /var/lib/apt/lists/*

# Note: do not install hf_transfer by default to avoid parallel-transfer limits

# Copy orchestrator, packages and entrypoint into image
COPY run.py /app/run.py
COPY env/ /app/env/
COPY util/ /app/util/
COPY env/entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
