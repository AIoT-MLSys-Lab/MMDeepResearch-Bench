"""Reason-aware validity penalty for MM-stage N/A.

This module is designed to be *additive* to an existing evaluation pipeline.
It does not change how per-sample scores are produced; instead, it provides a
post-hoc aggregation that penalizes models when the multimodal (MM/MOSAIC)
stage becomes unscorable ("N/A"), while distinguishing failure responsibility.

Core idea (MM-scoped):

  score_penalized = mean(score_valid) * (p_w_mm) ** alpha

where p_w_mm is a weighted validity rate computed over MM-attempted samples.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


BUCKET_MODEL = "model_failure"
BUCKET_PIPELINE = "pipeline_failure"
BUCKET_PROVIDER = "system_provider_failure"
BUCKET_DATA = "data_access_failure"
BUCKET_UNKNOWN = "unknown"


DEFAULT_BUCKET_WEIGHTS: Dict[str, float] = {
    BUCKET_MODEL: 0.0,
    BUCKET_PIPELINE: 0.5,
    BUCKET_PROVIDER: 0.8,
    BUCKET_DATA: 0.9,
    BUCKET_UNKNOWN: 0.5,  # conservative default
}


_NA_STRINGS = {"n/a", "na", "none", "null", "nan", ""}


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in _NA_STRINGS:
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _lower_join(*xs: Any) -> str:
    parts: List[str] = []
    for x in xs:
        if x is None:
            continue
        try:
            parts.append(str(x))
        except Exception:
            continue
    return "\n".join(parts).lower()


@dataclass(frozen=True)
class MMValidityConfig:
    """Configuration for MM-stage validity penalty."""

    alpha: float = 1.0
    bucket_weights: Mapping[str, float] = None  # type: ignore[assignment]
    # Keys used to locate per-sample values in heterogeneous record formats.
    mm_score_keys: Tuple[str, ...] = ("MM", "MOSAIC", "mm", "mm_score", "mosaic_score")
    mm_attempt_keys: Tuple[str, ...] = (
        "mm_triggered",
        "mosaic_triggered",
        "I_M",
        "I_MM",
        "mm_attempted",
    )
    final_score_keys: Tuple[str, ...] = (
        "FINAL_MMDR",
        "final_mmdr",
        "S_FINAL",
        "final_score",
        "score_final",
        "overall",
    )
    # Optional signal keys for better bucketing.
    report_len_keys: Tuple[str, ...] = ("report_chars", "report_len", "report_length", "md_chars")
    num_images_keys: Tuple[str, ...] = ("num_images", "image_count", "n_images")
    mm_items_keys: Tuple[str, ...] = ("mm_items", "mosaic_items", "num_mm_items")
    error_text_keys: Tuple[str, ...] = (
        "mm_error",
        "mosaic_error",
        "mm_exception",
        "exception",
        "error",
        "err",
        "traceback",
        "log",
    )
    http_code_keys: Tuple[str, ...] = ("http_code", "status_code", "code")

    def __post_init__(self) -> None:
        if self.bucket_weights is None:
            object.__setattr__(self, "bucket_weights", dict(DEFAULT_BUCKET_WEIGHTS))


def mm_attempted(record: Mapping[str, Any], cfg: MMValidityConfig) -> bool:
    """Whether MM/MOSAIC was intended/attempted for this record.

    Heuristic:
    - If explicit attempt flags exist, respect them.
    - Else, if an MM score field exists (even if N/A), treat it as attempted.
    """
    for k in cfg.mm_attempt_keys:
        if k in record:
            v = record.get(k)
            if isinstance(v, bool):
                return v
            vv = str(v).strip().lower()
            if vv in {"1", "true", "yes", "y"}:
                return True
            if vv in {"0", "false", "no", "n"}:
                return False
    # Fall back: presence of MM score key indicates the stage exists.
    return any(k in record for k in cfg.mm_score_keys)


def get_mm_score(record: Mapping[str, Any], cfg: MMValidityConfig) -> Optional[float]:
    for k in cfg.mm_score_keys:
        if k in record:
            v = _to_float(record.get(k))
            if v is not None:
                return v
            # If key exists but non-numeric (e.g., "N/A"), treat as missing.
            return None
    return None


def get_final_score(record: Mapping[str, Any], cfg: MMValidityConfig) -> Optional[float]:
    for k in cfg.final_score_keys:
        if k in record:
            v = _to_float(record.get(k))
            if v is not None:
                return v
            return None
    return None


def _get_first_int(record: Mapping[str, Any], keys: Tuple[str, ...]) -> Optional[int]:
    for k in keys:
        if k in record:
            try:
                return int(float(record.get(k)))
            except Exception:
                continue
    return None


def classify_mm_na_bucket(record: Mapping[str, Any], cfg: MMValidityConfig) -> str:
    """Assign a responsibility-aware bucket for MM-stage N/A.

    This is best-effort: it uses available error strings and lightweight
    metadata. If your pipeline already logs a structured reason, you can set
    record['mm_na_bucket'] and this function will respect it.
    """
    pre = (record.get("mm_na_bucket") or record.get("mosaic_na_bucket") or "").strip().lower()
    if pre:
        # Normalize a few common aliases.
        alias = {
            "model": BUCKET_MODEL,
            "model_failure": BUCKET_MODEL,
            "pipeline": BUCKET_PIPELINE,
            "pipeline_failure": BUCKET_PIPELINE,
            "system": BUCKET_PROVIDER,
            "provider": BUCKET_PROVIDER,
            "system_provider_failure": BUCKET_PROVIDER,
            "data": BUCKET_DATA,
            "data_access": BUCKET_DATA,
            "data_access_failure": BUCKET_DATA,
        }
        return alias.get(pre, pre)

    err_text = _lower_join(*[record.get(k) for k in cfg.error_text_keys])
    http_code = _get_first_int(record, cfg.http_code_keys)

    # System/provider failures.
    if http_code in {408, 425, 429, 500, 502, 503, 504}:
        return BUCKET_PROVIDER
    if any(x in err_text for x in [
        "rate limit", "ratelimit", "too many requests", "timeout", "timed out",
        "connection reset", "connection aborted", "service unavailable", "temporarily unavailable",
        "502", "503", "504", "429",
    ]):
        return BUCKET_PROVIDER

    # Data accessibility failures (common for MM: image 404 / anti-bot).
    if http_code in {401, 403, 404, 410}:
        return BUCKET_DATA
    if any(x in err_text for x in [
        "403", "404", "410", "forbidden", "not found", "gone", "blocked", "access denied",
        "captcha", "anti-bot", "antibot", "bot detection", "cloudflare", "robot check",
        "download failed", "image fetch", "failed to fetch image", "failed to download",
    ]):
        return BUCKET_DATA

    # Pipeline/tooling failures.
    if any(x in err_text for x in [
        "json", "schema", "parse", "parsing", "invalid argument", "typeerror", "keyerror",
        "valueerror", "filenotfounderror", "permissionerror", "traceback",
        "judge", "validator", "protobuf",
    ]):
        return BUCKET_PIPELINE

    # Model-attributable failures: empty/garbled report or no usable MM items.
    rep_len = _get_first_int(record, cfg.report_len_keys)
    if rep_len is not None and rep_len == 0:
        return BUCKET_MODEL
    mm_items = _get_first_int(record, cfg.mm_items_keys)
    num_images = _get_first_int(record, cfg.num_images_keys)
    if num_images is not None and num_images > 0 and mm_items is not None and mm_items == 0:
        return BUCKET_MODEL
    if any(x in err_text for x in [
        "empty report", "no output", "refused", "policy", "format violation", "missing references",
        "hallucinated", "nonsense", "garbled",
    ]):
        return BUCKET_MODEL

    return BUCKET_UNKNOWN


def mm_validity_weight(record: Mapping[str, Any], cfg: MMValidityConfig) -> Tuple[float, Optional[str]]:
    """Return (weight, bucket) for MM-stage validity.

    - If MM not attempted, we return (1.0, None) and exclude it from the MM
      validity denominator in aggregation.
    - If MM attempted and has a numeric score, return (1.0, None).
    - If MM attempted and is N/A, return (w(bucket), bucket).
    """
    if not mm_attempted(record, cfg):
        return 1.0, None

    if get_mm_score(record, cfg) is not None:
        return 1.0, None

    bucket = classify_mm_na_bucket(record, cfg)
    w = float(cfg.bucket_weights.get(bucket, cfg.bucket_weights.get(BUCKET_UNKNOWN, 0.5)))
    w = max(0.0, min(1.0, w))
    return w, bucket


def aggregate_with_mm_na_penalty(
    records: Iterable[Mapping[str, Any]],
    cfg: Optional[MMValidityConfig] = None,
) -> Dict[str, Any]:
    """Aggregate penalized score for a single model run.

    Expected input: an iterable of per-sample dicts.

    Returns a dict containing:
      - mean_valid_score
      - mm_weighted_validity (p_w_mm)
      - penalized_score
      - counts per bucket
      - attempted_mm_n
      - valid_score_n
    """
    cfg = cfg or MMValidityConfig()
    alpha = float(cfg.alpha)
    alpha = max(0.0, alpha)

    valid_scores: List[float] = []
    mm_weights: List[float] = []
    bucket_counts: Dict[str, int] = {k: 0 for k in cfg.bucket_weights.keys()}
    mm_attempt_n = 0

    for r in records:
        # Quality: use whatever the pipeline already computed as final.
        fs = get_final_score(r, cfg)
        if fs is not None:
            valid_scores.append(float(fs))

        w, bucket = mm_validity_weight(r, cfg)
        if bucket is not None or (mm_attempted(r, cfg) and get_mm_score(r, cfg) is None):
            # MM attempted.
            mm_attempt_n += 1
            mm_weights.append(float(w))
            if bucket is not None:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    mean_valid = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    p_w_mm = sum(mm_weights) / len(mm_weights) if mm_weights else 1.0
    penalized = mean_valid * (p_w_mm ** alpha)

    return {
        "mean_valid_score": mean_valid,
        "valid_score_n": len(valid_scores),
        "mm_weighted_validity": p_w_mm,
        "attempted_mm_n": mm_attempt_n,
        "alpha": alpha,
        "penalized_score": penalized,
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items() if v},
    }


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load records from JSON, JSONL, or a minimal CSV (header row required).

    - .json  : list[dict] or dict with a 'records' field
    - .jsonl : one dict per line
    - .csv   : comma-separated with simple values (no nested JSON)
    """
    p = path.lower()
    if p.endswith(".jsonl"):
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if p.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            return [dict(x) for x in obj["records"]]
        raise ValueError("Unsupported JSON structure: expected list[dict] or {records: [...]}")
    if p.endswith(".csv"):
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    raise ValueError("Unsupported file type (expected .json/.jsonl/.csv)")
