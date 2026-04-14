---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# Plant Disease Detection

Dual-model plant disease diagnosis: **MobileNetV3-Small CNN** (97.89% val accuracy) + **Gemini VLM** with confidence-weighted consensus.

Upload a plant leaf photo → get species, disease, severity, treatment, and prevention advice.

## Tech Stack

- **Frontend**: React 18 + Vite + Tailwind CSS
- **Backend**: FastAPI + PyTorch (MobileNetV3-Small)
- **VLM**: Google Gemini 2.5 Flash Lite (free tier)
- **Dataset**: PlantVillage — 38 classes, 87,000+ images

## Space Secrets Required

| Secret | Value |
|---|---|
| `MODEL_REPO_ID` | `your-hf-username/plant-disease-model` |
| `VLM_PROVIDER` | `gemini` |
| `GEMINI_API_KEY` | your Gemini API key |
| `HF_TOKEN` | only if model repo is private |

## Local Development

```bash
# Backend
cd backend
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend
cd "Plant Disease Detection"
cp .env.example .env
npm install
npm run dev
```

## Dataset

1. New Plant Diseases Dataset — https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. PlantVillage Dataset — https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
