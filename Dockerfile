# ─────────────────────────────────────────────────────────────
# Stage 1: Build the React / Vite frontend
# ─────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# Install dependencies first (cache layer)
COPY ["Plant Disease Detection/package.json", "Plant Disease Detection/package-lock.json", "./"]
RUN npm ci

# Copy source & inject production API URL (same origin as the backend)
COPY ["Plant Disease Detection/", "./"]
RUN echo "VITE_API_URL=" > .env.production
RUN npm run build
# Output: /frontend/dist


# ─────────────────────────────────────────────────────────────
# Stage 2: Python / FastAPI runtime
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Minimal system libs required by Pillow / OpenCV-free stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────
# Install CPU-only PyTorch first (saves ~1.7 GB vs CUDA wheel)
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────
COPY backend/ ./backend/

# ── Built frontend ───────────────────────────────────────────
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# ── Startup script ───────────────────────────────────────────
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["./start.sh"]
