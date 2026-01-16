from __future__ import annotations

import math
from typing import List


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        b = float(b)
        if b == 0.0:
            return default
        return float(a) / b
    except Exception:
        return default


def logistic(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5


def harmonic_mean(vals: List[float], eps: float = 1e-9) -> float:
    xs = [max(eps, float(v)) for v in vals if v is not None]
    if not xs:
        return 0.0
    return len(xs) / sum(1.0 / v for v in xs)


def softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps) if exps else 1.0
    return [e / s for e in exps]


def shannon_entropy(ps: List[float], eps: float = 1e-12) -> float:
    ps2 = [p for p in ps if p > eps]
    if not ps2:
        return 0.0
    return -sum(p * math.log(p + eps) for p in ps2)
