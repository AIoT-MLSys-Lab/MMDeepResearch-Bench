"""
Scoring entrypoint (thin wrapper).

This file is intentionally small. The full backbone implementation is split into
multiple modules:
- scoring_math.py
- scoring_text.py
- scoring_web.py
- scoring_sources.py
- scoring_golden.py
- scoring_judge.py
- scoring_general.py
- scoring_evidence.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .config import CRAWL
from .scoring_text import normalize_report_type
from .scoring_general import compute_general
from .scoring_evidence import compute_evidence


def score_report(
    report_text: str,
    rules_text: str,
    question_text: str,
    report_type: str = "daily_medium",
    golden_path: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    rt = normalize_report_type(report_type)
    qid = (question_id or "unknown").strip() or "unknown"
    t0 = time.time()

    gen = compute_general(report_text, rules_text, question_text, rt)

    patience_chars = (
        getattr(CRAWL, "patience_chars_research", 140_000)
        if rt == "research"
        else getattr(CRAWL, "patience_chars_daily", 90_000)
    )
    evi = compute_evidence(report_text, rules_text, question_text, rt, patience_chars=patience_chars)

    ctrl = gen.get("ctrl") or evi.get("ctrl") or {}
    lambda_hint = float(ctrl.get("lambda_final", 0.55 if "daily" in rt else 0.40))
    if "daily" in rt:
        lambda_hint = min(0.60, max(0.50, lambda_hint))
    if rt == "research":
        lambda_hint = min(0.45, max(0.35, lambda_hint))
    lambda_final = max(0.0, min(1.0, lambda_hint))

    FINAL = lambda_final * float(gen["score"]) + (1.0 - lambda_final) * float(evi["score"])

    golden_file = None
    if golden_path:
        if os.path.isdir(golden_path):
            golden_file = os.path.join(golden_path, f"{qid}.golden.json")
        else:
            golden_file = golden_path

        payload = {
            "question_id": qid,
            "report_type": rt,
            "question": question_text,
            "generated_at_unix": int(time.time()),
            "raw_pairs": evi.get("raw_pairs", []),
            "claims": evi.get("audited_claims", []),
            "general_judge": gen.get("judge", {}),
            "controller": ctrl,
        }
        try:
            os.makedirs(os.path.dirname(golden_file), exist_ok=True)
            with open(golden_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            golden_file = None

    out = {
        "question_id": qid,
        "report_type": rt,

        "GEN": round(float(gen["score"]), 3),
        "EVI": round(float(evi["score"]), 3),
        "FINAL": round(float(FINAL), 3),
        "lambda_final": round(float(lambda_final), 4),

        "general_score": round(float(gen["score"]), 3),
        "evidence_score": round(float(evi["score"]), 3),
        "final_score": round(float(FINAL), 3),

        "general": gen,
        "evidence": evi,
        "golden_json": golden_file,
        "timing_s": round(time.time() - t0, 3),
    }
    return out
