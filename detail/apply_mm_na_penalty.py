#!/usr/bin/env python3
"""Apply reason-aware MM N/A penalty to existing evaluation records.

Usage examples:
  python apply_mm_na_penalty.py --input results.jsonl
  python apply_mm_na_penalty.py --input results.jsonl --alpha 1 --out summary_penalized.json

The script groups records by model name if a model column exists; otherwise
it treats the input as a single model run.

It **does not** recompute any metric; it only rescales the final score using
MM-stage weighted validity.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

from .mm_na_penalty import (
    MMValidityConfig,
    aggregate_with_mm_na_penalty,
    load_records,
)


def _pick_model_name(rec: Dict[str, Any]) -> str:
    for k in ("model", "Model", "model_name", "report_model", "llm", "LLM"):
        if k in rec and rec.get(k):
            return str(rec.get(k))
    return "(single)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="records file (.json/.jsonl/.csv)")
    ap.add_argument("--alpha", type=float, default=1.0, help="penalty strength alpha")
    ap.add_argument("--out", default="", help="optional output JSON path")
    args = ap.parse_args()

    records = load_records(args.input)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[_pick_model_name(r)].append(r)

    cfg = MMValidityConfig(alpha=float(args.alpha))

    summary: Dict[str, Any] = {"alpha": cfg.alpha, "models": {}}
    for model, rs in groups.items():
        summary["models"][model] = aggregate_with_mm_na_penalty(rs, cfg)

    # Pretty print.
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
