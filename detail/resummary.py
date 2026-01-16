# filter_zero_final_mmdr.py
# Delete JSONL entries whose FINAL_MMDR == 0.0 (supports both top-level and nested: scoring.scores.FINAL_MMDR)

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def get_final_mmdr(obj: Dict[str, Any]) -> Optional[float]:
    # 1) top-level
    if "FINAL_MMDR" in obj:
        v = obj.get("FINAL_MMDR")
        try:
            return float(v)
        except Exception:
            return None

    # 2) common nested location in your pipeline
    cur = obj.get("scoring")
    if isinstance(cur, dict):
        cur2 = cur.get("scores")
        if isinstance(cur2, dict) and "FINAL_MMDR" in cur2:
            v = cur2.get("FINAL_MMDR")
            try:
                return float(v)
            except Exception:
                return None

    return None


def is_zero(v: Optional[float], eps: float = 1e-12) -> bool:
    if v is None:
        return False
    if math.isnan(v):
        return False
    return abs(v) <= eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    ap.add_argument("--drop-bad-json", action="store_true", help="Drop lines that are not valid JSON")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = dropped = bad = 0

    with in_path.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open("w", encoding="utf-8", newline="\n") as fout:
        for lineno, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                bad += 1
                if args.drop_bad_json:
                    continue
                # keep raw line if you don't want to lose anything
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                continue

            if not isinstance(obj, dict):
                # keep non-dict lines
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                continue

            fm = get_final_mmdr(obj)
            if is_zero(fm):
                dropped += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. total={total}, kept={kept}, dropped(FINAL_MMDR==0)={dropped}, bad_json={bad}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
