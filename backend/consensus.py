"""
Consensus Engine
Merges CNN probabilities and VLM predictions into a single
confidence-weighted diagnosis.

Weights:
  CNN  → 0.65  (task-specific, 97.3 % accuracy)
  VLM  → 0.35  (general vision-language reasoning)

If the CNN model is in demo mode (not loaded), VLM carries full weight.
"""

from typing import Dict, List

import numpy as np

from cnn_predictor import CLASS_NAMES, _parse_class

CNN_WEIGHT = 0.65
VLM_WEIGHT = 0.35


def _vlm_prob_vector(vlm_class: str | None, vlm_conf: float) -> np.ndarray:
    """
    Build a probability vector for the VLM prediction.
    The predicted class gets `vlm_conf`; the rest share the remaining mass uniformly.
    """
    n = len(CLASS_NAMES)
    vec = np.full(n, (1.0 - vlm_conf) / max(n - 1, 1))

    if vlm_class and vlm_class in CLASS_NAMES:
        idx = CLASS_NAMES.index(vlm_class)
        vec[idx] = vlm_conf  # override the predicted slot
    else:
        # VLM matched nothing — uniform distribution
        vec = np.ones(n) / n

    return vec


class ConsensusEngine:
    def compute(self, cnn_result: Dict, vlm_result: Dict) -> Dict:
        demo_mode: bool = cnn_result.get("demo_mode", False)
        cnn_w = 0.0 if demo_mode else CNN_WEIGHT
        vlm_w = 1.0 if demo_mode else VLM_WEIGHT

        # ── CNN probability vector ──────────────────────────────────────
        cnn_probs = np.array(cnn_result.get("all_probs", [0.0] * len(CLASS_NAMES)), dtype=float)
        if cnn_probs.sum() < 1e-9:
            cnn_probs = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)

        # ── VLM probability vector ──────────────────────────────────────
        vlm_class: str | None = vlm_result.get("matched_class")
        vlm_conf: float = float(vlm_result.get("confidence", 0.5))
        vlm_probs = _vlm_prob_vector(vlm_class, vlm_conf)

        # ── Weighted blend ─────────────────────────────────────────────
        consensus_probs = cnn_w * cnn_probs + vlm_w * vlm_probs
        total = consensus_probs.sum()
        if total > 0:
            consensus_probs /= total  # re-normalise

        # ── Final prediction ───────────────────────────────────────────
        final_idx: int = int(np.argmax(consensus_probs))
        final_class = CLASS_NAMES[final_idx]
        final_conf = float(consensus_probs[final_idx])
        final_plant, final_disease = _parse_class(final_class)

        # Top-5 from consensus for display
        top5_idxs = np.argsort(consensus_probs)[::-1][:5]
        top5: List[Dict] = [
            {
                "class": CLASS_NAMES[i],
                "plant": _parse_class(CLASS_NAMES[i])[0],
                "disease": _parse_class(CLASS_NAMES[i])[1],
                "confidence": float(consensus_probs[i]),
                "cnn_conf": float(cnn_probs[i]),
                "vlm_conf": float(vlm_probs[i]),
            }
            for i in top5_idxs
        ]

        # ── Agreement analysis ─────────────────────────────────────────
        cnn_top1 = cnn_result.get("top1_class", "")
        cnn_top5_classes = [x["class"] for x in cnn_result.get("top_k", [])]

        if vlm_class == cnn_top1:
            agreement = "strong"
            agreement_label = "Strong Agreement"
        elif vlm_class in cnn_top5_classes:
            agreement = "partial"
            agreement_label = "Partial Agreement"
        else:
            agreement = "conflict"
            agreement_label = "Models Disagree"

        return {
            "final_class": final_class,
            "final_plant": final_plant,
            "final_disease": final_disease,
            "final_confidence": final_conf,
            "agreement": agreement,
            "agreement_label": agreement_label,
            "top5": top5,
            "cnn_weight": round(cnn_w, 2),
            "vlm_weight": round(vlm_w, 2),
            "cnn_top1": cnn_top1,
            "vlm_prediction": vlm_class,
            "explanation": _build_explanation(
                agreement, final_plant, final_disease,
                cnn_result.get("top1_confidence", 0.0),
                vlm_conf, final_conf,
            ),
        }


def _build_explanation(
    agreement: str,
    plant: str,
    disease: str,
    cnn_conf: float,
    vlm_conf: float,
    final_conf: float,
) -> str:
    if agreement == "strong":
        return (
            f"Both models independently identified this as {plant} — {disease}. "
            f"The CNN (65% weight) reported {cnn_conf:.1%} confidence and the VLM "
            f"(35% weight) reported {vlm_conf:.1%}, yielding a consensus confidence "
            f"of {final_conf:.1%}."
        )
    if agreement == "partial":
        return (
            f"The CNN's top-1 prediction differs from the VLM, but the VLM's diagnosis "
            f"appears within the CNN's top-5. The weighted consensus favours "
            f"{plant} — {disease} at {final_conf:.1%} confidence. "
            f"Consider reviewing image quality or consulting a specialist."
        )
    return (
        f"The CNN and VLM reached different conclusions, which may indicate an unusual "
        f"disease presentation or ambiguous image. The weighted consensus suggests "
        f"{plant} — {disease} at {final_conf:.1%} confidence. "
        f"Expert consultation is strongly recommended."
    )
