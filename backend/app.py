"""
Plant Disease Diagnosis API
FastAPI backend that accepts a leaf image and returns:
  - CNN prediction (top-5 with confidence scores)
  - VLM diagnosis (Claude or GPT-4o visual analysis)
  - Confidence-weighted consensus

Usage:
    cd backend
    cp .env.example .env          # fill in your API key
    uvicorn app:app --reload --port 8000
"""

import base64
import io
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

load_dotenv()

# ── Lazy-import heavy modules after env is loaded ──────────────────────────
from cnn_predictor import CNNPredictor
from consensus import ConsensusEngine
from vlm_analyzer import VLMAnalyzer

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "../plant_disease__classification_model.pth")
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "anthropic")   # "anthropic" | "openai"

MAX_IMAGE_SIZE_MB = 10

# ──────────────────────────────────────────────────────────────────────────
# App & middleware
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Diagnosis API",
    version="1.0.0",
    description="Dual CNN + VLM plant disease diagnosis with confidence-weighted consensus.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "*",                       # remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# Component initialisation  (module-level so they load once at startup)
# ──────────────────────────────────────────────────────────────────────────
cnn_predictor = CNNPredictor(model_path=MODEL_PATH)
vlm_analyzer = VLMAnalyzer(provider=VLM_PROVIDER)
consensus_engine = ConsensusEngine()


# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    # In production the React build is at ../frontend/dist/index.html
    index = os.path.join(_FRONTEND_DIST, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "Plant Disease Diagnosis API is running.", "version": "1.0.0"}


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "cnn_loaded": cnn_predictor.is_loaded,
        "vlm_provider": VLM_PROVIDER,
        "model_path": MODEL_PATH,
    }


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    """
    Upload a plant leaf image.
    Returns CNN predictions, VLM analysis, and a consensus diagnosis.
    """
    # ── Validate content type ──────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()

    # ── Size guard ─────────────────────────────────────────────────────
    if len(raw_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_SIZE_MB} MB limit.",
        )

    # ── Decode image ───────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image. Ensure it is a valid JPEG/PNG.")

    # ── Base64 for VLM (re-encode as JPEG to normalise) ───────────────
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Run CNN ────────────────────────────────────────────────────────
    cnn_result = cnn_predictor.predict(image, topk=5)

    # ── Run VLM ────────────────────────────────────────────────────────
    try:
        vlm_result = await vlm_analyzer.analyze(img_b64, media_type="image/jpeg")
    except Exception as exc:
        vlm_result = {
            "plant_species": "Unknown",
            "disease_name": "VLM Error",
            "matched_class": None,
            "confidence": 0.0,
            "severity": "Unknown",
            "visual_symptoms": [],
            "diagnosis_explanation": f"VLM analysis failed: {str(exc)}",
            "treatment": {"immediate": [], "chemical": [], "biological": []},
            "prevention": [],
            "confidence_reasoning": "VLM call failed.",
            "model": VLM_PROVIDER,
            "error": True,
        }

    # ── Compute consensus ──────────────────────────────────────────────
    consensus = consensus_engine.compute(cnn_result, vlm_result)

    # Remove the large all_probs array from the response (not needed by UI)
    cnn_result.pop("all_probs", None)

    return JSONResponse(
        content={"cnn": cnn_result, "vlm": vlm_result, "consensus": consensus}
    )


# ──────────────────────────────────────────────────────────────────────────
# Serve React frontend (production / HF Spaces)
# Must be mounted LAST so API routes above take priority.
# ──────────────────────────────────────────────────────────────────────────
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
