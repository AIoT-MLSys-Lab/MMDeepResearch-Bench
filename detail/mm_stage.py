# mm_stage.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, DefaultDict, Tuple
from collections import defaultdict

from .mm_router import classify_mm_items, classify_image_type_from_text
from .pachong import resolve_images_from_items
from .mm_router3 import decide_engine, run_engine
from .judgers import judge_item


_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".svg")


def _avg(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _is_url(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")


def _extract_urls_from_any(x: Any) -> List[str]:
    out: List[str] = []
    if x is None:
        return out
    if isinstance(x, str):
        out += re.findall(r"https?://[^\s\)\"'>]+", x)
        return out
    if isinstance(x, dict):
        for _, v in x.items():
            out += _extract_urls_from_any(v)
        return out
    if isinstance(x, list):
        for it in x:
            out += _extract_urls_from_any(it)
        return out
    return out


def _infer_missing_reason(item: Dict[str, Any]) -> Tuple[str, str]:
    """把 404/403/反爬/无图等统一归一成“可访问性原因”。"""
    if not item.get("missing_image"):
        return ("ok", "")

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


def build_mm_items_from_golden(golden: Dict[str, Any], max_items: int = 12) -> List[Dict[str, Any]]:
    """Build MM items from golden claims.

    Two modes:
      - img_txt: one image vs one claim (existing behavior)
      - img_img: two images cited in the SAME sentence vs a comparative claim (new)

    We reserve a small budget for img_img items to avoid starving img_txt coverage.
    """
    claims = golden.get("claims") or []
    items: List[Dict[str, Any]] = []
    seen_single: set[str] = set()
    seen_pair: set[tuple[str, str, str]] = set()
    idx = 0

    max_pair_items = max(1, int(max_items // 4)) if max_items >= 4 else 0
    single_budget = max_items - max_pair_items

    def _is_image_seed(u: str) -> bool:
        uu = (u or "").lower()
        if uu.startswith("file://"):
            return True
        if os.path.exists(u):
            return True
        return uu.split("?", 1)[0].endswith(_IMG_EXT)

    def _looks_like_comparative_claim(text: str) -> bool:
        t = (text or "").lower()
        if any(k in t for k in ["compare", "contrast", "whereas", "while ", "unlike", "similar", "different", "both ", "respectively", "correspond"]):
            return True
        if any(k in (text or "") for k in ["对比", "相比", "相较", "相同", "一致", "不同", "差异", "分别", "前者", "后者", "两图", "两幅", "三图", "对照"]):
            return True
        return False

    pair_candidates: List[Dict[str, Any]] = []

    for c in claims:
        if len([x for x in items if x.get("mm_mode") == "img_txt"]) >= single_budget:
            break

        claim = (c.get("claim") or "").strip()
        url = (c.get("url") or "").strip()
        meta = c.get("evidence_meta")

        meta_urls = _extract_urls_from_any(meta)
        img_urls = [u for u in meta_urls if isinstance(u, str) and _is_image_seed(u)]
        img_urls = list(dict.fromkeys(img_urls))

        seeds: List[str] = []
        for u in img_urls[:2]:
            seeds.append(u)
        if url:
            seeds.append(url)

        seeds = [s for s in seeds if s]
        if not seeds:
            continue

        type_rule = classify_image_type_from_text(claim or "")
        if type_rule not in {"diagram", "datachart", "ocrchart", "photo", "imgpair"}:
            type_rule = "photo"

        for s in seeds:
            if len([x for x in items if x.get("mm_mode") == "img_txt"]) >= single_budget:
                break
            if s in seen_single:
                continue
            seen_single.add(s)

            items.append(
                {
                    "index": idx,
                    "type_rule": type_rule,
                    "type_final": type_rule,
                    "cls_source": "rule",
                    "caption": claim[:240] if claim else "",
                    "paragraphs": [claim] if claim else [],
                    "image_path": s,
                    "missing_image": False,
                    "source_url": url or s,
                    "source_file": golden.get("question_id", ""),
                    "mm_mode": "img_txt",
                }
            )
            idx += 1

        if max_pair_items <= 0:
            continue

        sent_urls = c.get("sentence_urls") or []
        if not isinstance(sent_urls, list) or len(sent_urls) < 2:
            continue
        if not _looks_like_comparative_claim(claim):
            continue

        img_sent = [u for u in sent_urls if isinstance(u, str) and _is_image_seed(u)]
        img_sent = list(dict.fromkeys(img_sent))
        if len(img_sent) < 2:
            continue

        top = img_sent[:3]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                key = (a, b, (claim or "")[:160])
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                pair_candidates.append(
                    {
                        "index": None,
                        "type_rule": "imgpair",
                        "type_final": "imgpair",
                        "cls_source": "pair",
                        "caption": (claim or "")[:240],
                        "paragraphs": [claim] if claim else [],
                        "image_path": a,
                        "image_path_b": b,
                        "missing_image": False,
                        "source_url": f"{a}||{b}",
                        "source_file": golden.get("question_id", ""),
                        "mm_mode": "img_img",
                    }
                )

    if max_pair_items > 0 and pair_candidates:
        for it in pair_candidates:
            if len(items) >= max_items:
                break
            current_pairs = sum(1 for x in items if x.get("mm_mode") == "img_img")
            if current_pairs >= max_pair_items:
                break
            it["index"] = idx
            items.append(it)
            idx += 1

    return items

def _l5_aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _safe_overall_score(item: Dict[str, Any]) -> Optional[float]:
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

    n_total = len(items)
    n_accessible = 0
    n_scored = 0

    scores_all: List[float] = []
    scores_by_type: DefaultDict[str, List[float]] = defaultdict(list)
    counts_by_type_total: DefaultDict[str, int] = defaultdict(int)
    counts_by_type_accessible: DefaultDict[str, int] = defaultdict(int)
    scores_by_judge: DefaultDict[str, List[float]] = defaultdict(list)
    dim_metrics_all: DefaultDict[str, List[float]] = defaultdict(list)
    missing_reason_counts: DefaultDict[str, int] = defaultdict(int)

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

    avg01 = _avg(scores_all)

    summary = {
        "num_items_total": n_total,
        "num_items_accessible": n_accessible,
        "num_items_scored": n_scored,
        "mm_coverage_accessible": (None if n_total == 0 else round(n_accessible / max(1, n_total), 6)),
        "mm_coverage_scored": (None if n_total == 0 else round(n_scored / max(1, n_total), 6)),
        "num_items_by_type_total": dict(counts_by_type_total),
        "num_items_by_type_accessible": dict(counts_by_type_accessible),
        "missing_reason_counts": dict(missing_reason_counts),
        "avg_score_overall": avg01,
        "avg_score_by_type": {t: _avg(v) for t, v in scores_by_type.items()},
        "avg_score_by_judge": {j: _avg(v) for j, v in scores_by_judge.items()},
        "avg_metric_by_dim": {d: _avg(v) for d, v in dim_metrics_all.items()},
    }
    return {"summary": summary, "items": items}


def run_mm_stage_from_golden(
    golden_path: str,
    work_dir: str,
    max_items: int = 12,
    dump_items_json: Optional[str] = None,
) -> Dict[str, Any]:
    golden_path = str(golden_path)
    work_dir = str(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    items = build_mm_items_from_golden(golden, max_items=max_items)
    if not items:
        return {
            "activated": True,
            "mm_score": None,
            "mm_score_01": None,
            "reason": "no_items_from_golden",
            "num_items": 0,
            "num_items_accessible": 0,
            "num_items_scored": 0,
        }

    items = classify_mm_items(items, use_clip=False, use_gemini=True, prefer="gpt")

    img_dir = os.path.join(work_dir, "images")
    items = resolve_images_from_items(items, out_dir=img_dir)

    for it in items:
        # Special handling for image-image (pair) items: require BOTH images.
        if (it.get("mm_mode") == "img_img") or ((it.get("type_final") or "").lower() == "imgpair"):
            locs = it.get("local_images")
            if isinstance(locs, list) and len(locs) >= 2 and all(p and os.path.exists(str(p)) for p in locs[:2]):
                it["image_path"] = str(locs[0])
                it["image_path_b"] = str(locs[1])
                it["local_image"] = str(locs[0])
                it["missing_image"] = False
                it["missing_reason"] = "ok"
                it["missing_detail"] = ""
            else:
                it["missing_image"] = True
                it["missing_reason"] = "pair_images_unresolved"
                it["missing_detail"] = f"local_images={locs}"
            continue

        local = it.get("local_image")
        if local and os.path.exists(local):
            it["image_path"] = local
            it["missing_image"] = False
            it["missing_reason"] = "ok"
            it["missing_detail"] = ""
        else:
            it["missing_image"] = True
            r, detail = _infer_missing_reason(it)
            it["missing_reason"] = r
            it["missing_detail"] = detail

    items = classify_mm_items(items, use_clip=True, use_gemini=True, prefer="clip")

    for it in items:
        tf = (it.get("type_final") or it.get("type") or "photo").lower()
        if tf not in {"diagram", "datachart", "ocrchart", "photo", "imgpair"}:
            it["type_final"] = "photo"

    for it in items:
        if it.get("missing_image"):
            it["engine"] = "skipped"
            it["engine_output"] = {
                "engine_name": "skipped",
                "mode": "visual_eval",
                "metrics": {},
                "answer": "skipped_missing_image",
            }
            continue

        eng = decide_engine(it)
        it["engine"] = eng
        try:
            it["engine_output"] = run_engine(eng, it)
        except Exception as e:
            it["engine_error"] = str(e)
            it["engine_output"] = {
                "engine_name": eng,
                "mode": "visual_eval",
                "metrics": {},
                "answer": "engine_error",
            }

    for it in items:
        try:
            it["judge"] = judge_item(it)
        except Exception as e:
            it["judge_error"] = str(e)
            it["judge"] = {
                "judge_name": "judge_error",
                "score_overall": None,
                "metrics": {},
                "weights": {},
                "meta": {"skipped": True, "skip_reason": "judge_error"},
            }

    if dump_items_json:
        Path(os.path.dirname(dump_items_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(dump_items_json, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    routed_dir = os.path.join(work_dir, "mm_router_routed")
    os.makedirs(routed_dir, exist_ok=True)
    all_with_scores = os.path.join(routed_dir, "all_with_scores.json")
    with open(all_with_scores, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    final_obj = _l5_aggregate(items)
    final_summary_path = os.path.join(routed_dir, "final_summary.json")
    with open(final_summary_path, "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)

    avg01 = final_obj["summary"]["avg_score_overall"]
    mm_score = None if avg01 is None else round(float(avg01) * 100.0, 3)

    return {
        "activated": True,
        "mm_score": mm_score,
        "mm_score_01": avg01,
        "num_items": len(items),
        "num_items_accessible": int(final_obj["summary"]["num_items_accessible"]),
        "num_items_scored": int(final_obj["summary"]["num_items_scored"]),
        "coverage_accessible": final_obj["summary"]["mm_coverage_accessible"],
        "coverage_scored": final_obj["summary"]["mm_coverage_scored"],
        "missing_reason_counts": final_obj["summary"].get("missing_reason_counts", {}),
        "items_path": dump_items_json,
        "l5_final_summary": final_summary_path,
        "l4_all_with_scores": all_with_scores,
    }
