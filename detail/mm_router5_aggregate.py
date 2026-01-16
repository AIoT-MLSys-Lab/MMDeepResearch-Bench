# mm_router5_aggregate.py
from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any, DefaultDict, Tuple
from collections import defaultdict


def _safe_overall_score(item: Dict[str, Any]) -> float | None:
    judge = item.get("judge")
    if not isinstance(judge, dict):
        return None
    v = judge.get("score_overall")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_dim_metrics(item: Dict[str, Any]) -> Dict[str, float]:
    judge = item.get("judge") or {}
    m = judge.get("metrics") or {}
    if not isinstance(m, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in m.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _infer_missing_reason(item: Dict[str, Any]) -> Tuple[str, str]:
    if not item.get("missing_image"):
        return ("ok", "")

    mr = str(item.get("missing_reason") or "").strip()
    md = str(item.get("missing_detail") or "").strip()
    if mr:
        return (mr, md)

    pfe = str(item.get("page_fetch_error") or "").lower()
    lde = str(item.get("last_download_error") or "").lower()
    de = str(item.get("download_error") or "").lower()
    msg = pfe or lde or de

    if "404" in msg or "410" in msg or "not found" in msg:
        return ("dead_link", msg)
    if "403" in msg or "401" in msg or "forbidden" in msg or "unauthorized" in msg:
        return ("blocked", msg)
    if "not_image_content_type" in msg or "text/html" in msg:
        return ("not_image", msg)
    if "too_small" in msg:
        return ("too_small_or_placeholder", msg)
    if int(item.get("num_candidates") or 0) == 0 and (item.get("page_fetch_error") is None):
        return ("no_image_found", msg)
    return ("download_failed", msg)


def _avg(xs: List[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def main():
    parser = argparse.ArgumentParser(
        description="L5：根据 all_with_scores.json 做文档级 final score 汇总（严谨口径：缺图不计分）"
    )
    parser.add_argument(
        "--in_json",
        default=os.path.join("mm_router_routed", "all_with_scores.json"),
        help="L4 输出的 all_with_scores.json（默认: mm_router_routed/all_with_scores.json）",
    )
    parser.add_argument(
        "--out_json",
        default=os.path.join("mm_router_routed", "final_summary.json"),
        help="L5 汇总输出路径（默认: mm_router_routed/final_summary.json）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_json):
        raise FileNotFoundError(args.in_json)

    with open(args.in_json, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    n_total = len(items)

    scores_all: List[float] = []
    scores_by_type: DefaultDict[str, List[float]] = defaultdict(list)
    counts_by_type_total: DefaultDict[str, int] = defaultdict(int)
    counts_by_type_accessible: DefaultDict[str, int] = defaultdict(int)
    scores_by_judge: DefaultDict[str, List[float]] = defaultdict(list)
    dim_metrics_all: DefaultDict[str, List[float]] = defaultdict(list)
    missing_reason_counts: DefaultDict[str, int] = defaultdict(int)

    n_accessible = 0
    n_scored = 0

    for it in items:
        t = (it.get("type_final") or it.get("type") or "photo").lower()
        counts_by_type_total[t] += 1

        if bool(it.get("missing_image")):
            r, _ = _infer_missing_reason(it)
            missing_reason_counts[r] += 1
            continue

        n_accessible += 1
        counts_by_type_accessible[t] += 1

        judge = it.get("judge") or {}
        judge_name = str(judge.get("judge_name", "unknown"))

        overall = _safe_overall_score(it)
        if overall is not None:
            n_scored += 1
            scores_all.append(overall)
            scores_by_type[t].append(overall)
            scores_by_judge[judge_name].append(overall)

        dims = _safe_dim_metrics(it)
        for k, v in dims.items():
            dim_metrics_all[k].append(v)

    summary: Dict[str, Any] = {}
    summary["num_items_total"] = n_total
    summary["num_items_accessible"] = n_accessible
    summary["num_items_scored"] = n_scored
    summary["mm_coverage_accessible"] = None if n_total == 0 else round(n_accessible / max(1, n_total), 6)
    summary["mm_coverage_scored"] = None if n_total == 0 else round(n_scored / max(1, n_total), 6)
    summary["num_items_by_type_total"] = dict(counts_by_type_total)
    summary["num_items_by_type_accessible"] = dict(counts_by_type_accessible)
    summary["missing_reason_counts"] = dict(missing_reason_counts)

    summary["avg_score_overall"] = _avg(scores_all)
    summary["avg_score_by_type"] = {t: _avg(v) for t, v in scores_by_type.items()}
    summary["avg_score_by_judge"] = {j: _avg(v) for j, v in scores_by_judge.items()}
    summary["avg_metric_by_dim"] = {d: _avg(v) for d, v in dim_metrics_all.items()}

    out_obj = {"summary": summary, "items": items}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("====== L5 Final Summary (Strict) ======")
    print(f"总 items: {n_total}")
    print(f"可访问 items: {n_accessible} (coverage={summary['mm_coverage_accessible']})")
    print(f"有分 items: {n_scored} (coverage={summary['mm_coverage_scored']})")
    print(f"整体平均分(01): {summary['avg_score_overall']}")
    if missing_reason_counts:
        print("missing 原因分布:")
        for k, v in sorted(missing_reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k:26s}: {v}")
    print("=" * 40)
    print(f"L5 汇总已写入: {args.out_json}")


if __name__ == "__main__":
    main()
