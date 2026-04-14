#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Plant Disease Detection — HF Spaces startup script
# Downloads the model checkpoint from HF Hub, then starts
# the FastAPI server on port 7860.
# ─────────────────────────────────────────────────────────────
set -e

MODEL_DIR="/app/model"
MODEL_FILE="plant_disease__classification_model.pth"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

mkdir -p "$MODEL_DIR"

# ── Download model from HF Hub if not already present ────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "[startup] Downloading model from Hugging Face Hub…"
    python - <<'PYEOF'
import os, sys, shutil
from huggingface_hub import hf_hub_download

repo_id  = os.environ.get("MODEL_REPO_ID")
token    = os.environ.get("HF_TOKEN", None)
filename = "plant_disease__classification_model.pth"
dest     = f"/app/model/{filename}"

if not repo_id:
    print("[ERROR] MODEL_REPO_ID environment variable is not set.")
    print("        Set it to your HF model repo, e.g.  username/plant-disease-model")
    sys.exit(1)

cached = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=token,
    cache_dir="/tmp/hf_cache",
)
shutil.copy(cached, dest)
print(f"[startup] Model saved to {dest}")
PYEOF
else
    echo "[startup] Model already present at $MODEL_PATH — skipping download."
fi

# ── Export env vars for the FastAPI app ──────────────────────
export MODEL_PATH="$MODEL_PATH"

# ── Launch FastAPI ────────────────────────────────────────────
echo "[startup] Starting FastAPI on port 7860…"
cd /app/backend
exec uvicorn app:app --host 0.0.0.0 --port 7860
