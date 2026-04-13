"""
VLM Analyzer
Sends a plant leaf image to a vision-language model and returns a
structured plant disease diagnosis.

Supported providers (set VLM_PROVIDER in .env):
  "gemini"    → Google Gemini 1.5 Flash  — FREE tier, no card needed
  "ollama"    → LLaVA running locally via Ollama — fully free, offline
  "anthropic" → Claude claude-sonnet-4-6  — paid
  "openai"    → GPT-4o              — paid
"""

import asyncio
import base64
import json
import os
from typing import Dict

from cnn_predictor import CLASS_NAMES

# ──────────────────────────────────────────────
# Shared prompt text
# ──────────────────────────────────────────────
_CLASS_LIST = "\n".join(f"  - {c}" for c in CLASS_NAMES)

SYSTEM_PROMPT = (
    "You are an expert plant pathologist and agronomist with decades of experience "
    "diagnosing plant diseases from visual leaf symptoms. You respond exclusively in "
    "valid JSON — no markdown, no prose outside the JSON object."
)

USER_PROMPT = f"""Carefully examine this plant leaf image and provide a precise disease diagnosis.

Respond with ONLY a valid JSON object matching this schema exactly:

{{
  "plant_species": "<common plant name, e.g. Tomato>",
  "disease_name": "<disease name, or 'Healthy' if no disease>",
  "matched_class": "<exact string from the class list below>",
  "confidence": <float 0.0-1.0>,
  "severity": "<None | Mild | Moderate | Severe>",
  "visual_symptoms": ["<symptom 1>", "<symptom 2>", "<symptom 3>"],
  "diagnosis_explanation": "<2-3 sentence clinical explanation of observed symptoms and reasoning>",
  "treatment": {{
    "immediate": ["<action>"],
    "chemical": ["<fungicide / pesticide with application rate>"],
    "biological": ["<organic or biological control>"]
  }},
  "prevention": ["<tip 1>", "<tip 2>", "<tip 3>"],
  "confidence_reasoning": "<brief explanation of your confidence level>"
}}

Supported classes (pick the single best match for matched_class):
{_CLASS_LIST}

Rules:
- matched_class MUST be copied verbatim from the list above.
- If the leaf appears healthy, use the appropriate *___healthy class.
- confidence should reflect genuine diagnostic certainty (0.0 = unsure, 1.0 = certain).
- Output ONLY the JSON object — no other text.
"""

COMBINED_PROMPT = SYSTEM_PROMPT + "\n\n" + USER_PROMPT  # for models with no system role


# ──────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────
class VLMAnalyzer:
    """
    Provider routing:
      gemini    → google-generativeai SDK  (FREE)
      ollama    → local HTTP to :11434     (FREE, offline)
      anthropic → anthropic SDK            (paid)
      openai    → openai SDK               (paid)
    """

    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()
        self._client = self._build_client()

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. "
                    "Get a free key at https://aistudio.google.com/app/apikey"
                )
            # Client is created per-call in _gemini() using the new google-genai SDK
            return None

        if self.provider == "ollama":
            # No client object needed; we call the REST API directly
            return None

        if self.provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
            return anthropic.Anthropic(api_key=api_key)

        if self.provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            return openai.AsyncOpenAI(api_key=api_key)

        raise ValueError(
            f"Unknown VLM provider: '{self.provider}'. "
            "Choose gemini | ollama | anthropic | openai"
        )

    # ------------------------------------------------------------------
    async def analyze(self, image_b64: str, media_type: str = "image/jpeg") -> Dict:
        dispatch = {
            "gemini":    self._gemini,
            "ollama":    self._ollama,
            "anthropic": self._claude,
            "openai":    self._gpt4o,
        }
        return await dispatch[self.provider](image_b64, media_type)

    # ══════════════════════════════════════════════════════════════════
    # FREE — Google Gemini 1.5 Flash  (uses new google-genai SDK)
    # Free tier: 1,500 req/day · 15 RPM · no billing required
    # Get key: https://aistudio.google.com/app/apikey
    # ══════════════════════════════════════════════════════════════════
    async def _gemini(self, image_b64: str, media_type: str) -> Dict:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_b64),
            mime_type=media_type,
        )

        def _sync_call():
            return client.models.generate_content(
                model=model_name,
                contents=[image_part, COMBINED_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                ),
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _sync_call)
        raw = response.text.strip()
        return self._parse(raw, model_label=f"{model_name} Vision (free)")

    # ══════════════════════════════════════════════════════════════════
    # FREE — Ollama + LLaVA  (fully local, no internet needed)
    # Install: https://ollama.com  →  ollama pull llava
    # ══════════════════════════════════════════════════════════════════
    async def _ollama(self, image_b64: str, media_type: str) -> Dict:
        import aiohttp

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llava")

        payload = {
            "model": ollama_model,
            "prompt": COMBINED_PROMPT,
            "images": [image_b64],   # Ollama accepts raw base64
            "stream": False,
            "options": {"temperature": 0.1},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text[:200]}")
                data = await resp.json()

        raw = data.get("response", "").strip()
        return self._parse(raw, model_label=f"Ollama {ollama_model} (local, free)")

    # ══════════════════════════════════════════════════════════════════
    # PAID — Anthropic Claude
    # ══════════════════════════════════════════════════════════════════
    async def _claude(self, image_b64: str, media_type: str) -> Dict:
        def _sync_call():
            return self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }],
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _sync_call)
        raw = response.content[0].text.strip()
        return self._parse(raw, model_label="Claude claude-sonnet-4-6 Vision")

    # ══════════════════════════════════════════════════════════════════
    # PAID — OpenAI GPT-4o
    # ══════════════════════════════════════════════════════════════════
    async def _gpt4o(self, image_b64: str, media_type: str) -> Dict:
        data_url = f"data:{media_type};base64,{image_b64}"
        response = await self._client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content.strip()
        return self._parse(raw, model_label="GPT-4o Vision")

    # ══════════════════════════════════════════════════════════════════
    # Shared response parser
    # ══════════════════════════════════════════════════════════════════
    def _parse(self, raw: str, model_label: str) -> Dict:
        text = raw
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            data: Dict = json.loads(text)
        except json.JSONDecodeError as exc:
            return {
                "plant_species": "Unknown",
                "disease_name": "Parse error",
                "matched_class": None,
                "confidence": 0.0,
                "severity": "Unknown",
                "visual_symptoms": [],
                "diagnosis_explanation": f"VLM response could not be parsed: {exc}",
                "treatment": {"immediate": [], "chemical": [], "biological": []},
                "prevention": [],
                "confidence_reasoning": "Parsing failed.",
                "model": model_label,
                "raw_response": raw[:500],
                "error": True,
            }

        data["confidence"] = float(data.get("confidence", 0.7))
        data.setdefault("treatment", {"immediate": [], "chemical": [], "biological": []})
        data.setdefault("visual_symptoms", [])
        data.setdefault("prevention", [])
        data["model"] = model_label
        return data
