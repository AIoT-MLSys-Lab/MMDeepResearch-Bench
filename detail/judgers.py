# judgers.py
from __future__ import annotations

from typing import Dict, Any


TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "photo": {"semantic": 0.6, "faithful": 0.25, "formatting": 0.15},
    "imgpair": {"semantic": 0.6, "faithful": 0.25, "formatting": 0.15},
    "diagram": {"semantic": 0.5, "faithful": 0.4, "formatting": 0.1},
    "datachart": {"semantic": 0.4, "faithful": 0.5, "formatting": 0.1},
    "ocrchart": {"semantic": 0.4, "faithful": 0.4, "formatting": 0.2},
    "default": {"semantic": 0.5, "faithful": 0.3, "formatting": 0.2},
}


def _get_metrics_from_engine(item: Dict[str, Any]) -> Dict[str, float]:
    eo = item.get("engine_output") or {}
    m = eo.get("metrics") or {}
    if not isinstance(m, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in m.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _weighted_sum(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    w_sum = 0.0
    for k, w in weights.items():
        if w <= 0:
            continue
        v = metrics.get(k)
        if v is None:
            v = 0.0
        s += float(v) * float(w)
        w_sum += float(w)
    if w_sum <= 0:
        return 0.0
    return s / w_sum


def judge_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    严谨口径（论文友好）：
      - missing_image=True 的 item：不产生分数（score_overall=None），不把 formatting 当“保底分”
      - 聚合层只统计 score_overall 有值的 item
    """
    t = (item.get("type_final") or item.get("type") or "photo").lower()
    judge_name = f"{t}_mm_judge_v0"
    eo = item.get("engine_output") or {}

    if bool(item.get("missing_image")):
        return {
            "judge_name": judge_name,
            "score_overall": None,  # 关键：缺图不计分
            "metrics": {},
            "weights": TYPE_WEIGHTS.get(t, TYPE_WEIGHTS["default"]),
            "meta": {
                "engine_name": eo.get("engine_name"),
                "skipped": True,
                "skip_reason": "missing_image",
            },
        }

    metrics = _get_metrics_from_engine(item)
    weights = TYPE_WEIGHTS.get(t, TYPE_WEIGHTS["default"])
    overall = _weighted_sum(metrics, weights)

    return {
        "judge_name": judge_name,
        "score_overall": float(overall),
        "metrics": metrics,
        "weights": weights,
        "meta": {"engine_name": eo.get("engine_name")},
    }
