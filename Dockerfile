# ==============================================================================
# KINETIX-AI — Production Docker Container & Hugging Face Spaces Engine
# Developer: Nicolas Cardozo | Pontificia Universidad Javeriana
# ==============================================================================

FROM python:3.11-slim

# System level libraries required for OpenCV headless & MediaPipe rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Pre-download MediaPipe Heavy task file into /app/data/models
RUN python -c "from src.models.pose_detector import PoseDetector; p = PoseDetector(); p.close()"

# Set permissions for unprivileged execution (standard for HF Spaces and cloud hosting)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose standard port (7860 default on Hugging Face, 8000 on standard servers)
ENV PORT=7860
ENV HOST=0.0.0.0
EXPOSE 7860

# Run FastAPI production server
CMD ["python", "app.py"]
