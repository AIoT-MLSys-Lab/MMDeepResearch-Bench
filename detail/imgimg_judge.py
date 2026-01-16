# imgimg_judge.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from .config import MM

try:
    from .clip_classifier import (
        ClipNotAvailable,
        get_image_embedding,
        cosine_similarity01,
    )
except Exception:  # pragma: no cover
    ClipNotAvailable = Exception  # type: ignore
    get_image_embedding = None  # type: ignore
    cosine_similarity01 = None  # type: ignore

try:
    from .api import gemini_mm_json
except Exception:  # pragma: no cover
    gemini_mm_json = None  # type: ignore


@dataclass
class PairJudgeResult:
    pair: Tuple[int, int]
    sim01: Optional[float]
    relation: str
    consistency_score: float
    confidence: float
    notes: str = ""


_PAIR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["support", "contradict", "unrelated", "uncertain"],
        },
        "consistency_score": {"type": "number"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["relation", "consistency_score", "confidence"],
}


def _clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _pack_context(it: Dict[str, Any], max_chars: int = 480) -> str:
    cap = (it.get("caption") or "").strip()
    paras = it.get("paragraphs") or []
    if isinstance(paras, list):
        p0 = str(paras[0]) if paras else ""
    else:
        p0 = str(paras)
    txt = (cap + "\n" + p0).strip()
    if len(txt) > max_chars:
        txt = txt[: max_chars - 3] + "..."
    return txt


def _group_key(it: Dict[str, Any]) -> str:
    if (MM.img_img_grouping or "").lower() == "context":
        cid = it.get("context_id") or it.get("paragraph_id") or it.get("context_index")
        if cid is not None and str(cid).strip() != "":
            return str(cid)
    return "__all__"


def _collect_accessible(items: List[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    idxs: List[int] = []
    paths: List[str] = []
    for i, it in enumerate(items):
        if bool(it.get("missing_image")):
            continue
        p = it.get("image_path") or it.get("local_image")
        if isinstance(p, str) and os.path.exists(p):
            idxs.append(i)
            paths.append(p)
    return idxs, paths


def _select_candidate_pairs(
    items: List[Dict[str, Any]],
    idxs: List[int],
    embs: Dict[int, Any],
) -> List[Tuple[int, int]]:
    """Select candidate pairs for LLM judging.

    Preference order:
      1) Within-group topk_per_group by CLIP sim (>= min_sim)
      2) Global topk by CLIP sim (>= min_sim)
      3) If no CLIP embeddings, fall back to sequential pairs within groups
    """
    topk = max(0, int(MM.img_img_topk))
    if topk <= 0:
        return []

    topk_g = max(0, int(MM.img_img_topk_per_group))
    min_sim = float(MM.img_img_min_sim)

    # ---- Fallback: no CLIP ----
    if not embs or cosine_similarity01 is None:
        by_g: DefaultDict[str, List[int]] = defaultdict(list)
        for i in idxs:
            by_g[_group_key(items[i])].append(i)
        pairs: List[Tuple[int, int]] = []
        for g, ids in by_g.items():
            for a, b in zip(ids, ids[1:]):
                pairs.append((a, b))
                if len(pairs) >= topk:
                    return pairs
        return pairs[:topk]

    # ---- CLIP-based selection ----
    pairs_set = set()
    scored_pairs: List[Tuple[float, Tuple[int, int]]] = []

    # per-group
    by_g: DefaultDict[str, List[int]] = defaultdict(list)
    for i in idxs:
        by_g[_group_key(items[i])].append(i)
    for g, ids in by_g.items():
        if len(ids) < 2:
            continue
        tmp: List[Tuple[float, Tuple[int, int]]] = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                sim = cosine_similarity01(embs[a], embs[b])
                if sim < min_sim:
                    continue
                tmp.append((sim, (a, b)))
        tmp.sort(key=lambda t: t[0], reverse=True)
        for sim, pr in tmp[:topk_g]:
            if pr not in pairs_set and (pr[1], pr[0]) not in pairs_set:
                pairs_set.add(pr)
                scored_pairs.append((sim, pr))

    # global fill
    if len(scored_pairs) < topk:
        tmp2: List[Tuple[float, Tuple[int, int]]] = []
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                a, b = idxs[x], idxs[y]
                pr = (a, b)
                if pr in pairs_set or (b, a) in pairs_set:
                    continue
                sim = cosine_similarity01(embs[a], embs[b])
                if sim < min_sim:
                    continue
                tmp2.append((sim, pr))
        tmp2.sort(key=lambda t: t[0], reverse=True)
        for sim, pr in tmp2:
            if len(scored_pairs) >= topk:
                break
            pairs_set.add(pr)
            scored_pairs.append((sim, pr))

    # final
    scored_pairs.sort(key=lambda t: t[0], reverse=True)
    return [pr for _, pr in scored_pairs[:topk]]


def _llm_pair_judge(
    img_a: str,
    img_b: str,
    ctx_a: str,
    ctx_b: str,
    sim01: Optional[float],
) -> PairJudgeResult:
    """Call multimodal LLM to judge whether two images are consistent.

    Returns a PairJudgeResult; on failure returns an 'uncertain' with score 0.5.
    """
    # conservative fallback
    fallback = PairJudgeResult(
        pair=(-1, -1),
        sim01=sim01,
        relation="uncertain",
        consistency_score=0.5,
        confidence=0.0,
        notes="llm_unavailable",
    )

    if not MM.img_img_use_llm_judge:
        return fallback
    if gemini_mm_json is None:
        return fallback

    sys = (
        "You are an evaluator for a multimodal research benchmark. "
        "Given TWO figures from the SAME report, decide whether they are mutually consistent "
        "with respect to the claims they are used to support. "
        "If they imply conflicting quantitative values, trends, labels, or conclusions, mark 'contradict'. "
        "If they support the same conclusion or are coherent variants (same plot/scene), mark 'support'. "
        "If they are about different things or cannot be compared, mark 'unrelated' or 'uncertain'. "
        "Output ONLY JSON that matches the schema."
    )
    user = (
        "Figure A context (caption + nearby paragraph):\n"
        f"{ctx_a}\n\n"
        "Figure B context (caption + nearby paragraph):\n"
        f"{ctx_b}\n\n"
        f"CLIP similarity estimate (0..1, optional): {sim01 if sim01 is not None else 'N/A'}\n\n"
        "Return JSON fields: relation, consistency_score (0..1), confidence (0..1), notes (short). "
        "Guide for consistency_score:\n"
        "- support: 0.85~1.0\n"
        "- unrelated: ~0.6 (do not over-penalize if incomparable)\n"
        "- uncertain: ~0.5\n"
        "- contradict: 0.0~0.2\n"
    )

    txt = gemini_mm_json(
        image_paths=[img_a, img_b],
        prompt=user,
        json_schema=_PAIR_SCHEMA,
        model=MM.img_img_llm_model,
        system_instruction=sys,
        temperature=float(MM.img_img_llm_temperature),
        max_output_tokens=int(MM.img_img_llm_max_tokens),
    )

    try:
        obj = json.loads(txt)
        relation = str(obj.get("relation") or "uncertain").strip().lower()
        if relation not in {"support", "contradict", "unrelated", "uncertain"}:
            relation = "uncertain"
        score = _clamp01(obj.get("consistency_score"), default=0.5)
        conf = _clamp01(obj.get("confidence"), default=0.5)
        notes = str(obj.get("notes") or "").strip()
        return PairJudgeResult(
            pair=(-1, -1),
            sim01=sim01,
            relation=relation,
            consistency_score=score,
            confidence=conf,
            notes=notes,
        )
    except Exception:
        return fallback


def judge_image_image_consistency(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a semantic image-image consistency score using a multimodal LLM.

    Returns dict:
      {
        "available": bool,
        "mean01": float | None,
        "n_pairs": int,
        "label_counts": {...},
        "pairs": [ {a_idx,b_idx,sim01,relation,score,confidence,notes} ... ],
        "fallback_used": "llm"|"clip"|"none",
      }

    This module does NOT decide how to blend into the final MM score.
    """
    idxs, _paths = _collect_accessible(items)
    if len(idxs) < 2:
        return {
            "available": False,
            "mean01": None,
            "n_pairs": 0,
            "label_counts": {},
            "pairs": [],
            "fallback_used": "none",
        }

    # precompute CLIP embeddings if possible (for pair selection and similarity in outputs)
    embs: Dict[int, Any] = {}
    if get_image_embedding is not None:
        for i in idxs:
            p = items[i].get("image_path") or items[i].get("local_image")
            if not isinstance(p, str) or not os.path.exists(p):
                continue
            try:
                embs[i] = get_image_embedding(p)
            except ClipNotAvailable:
                embs = {}
                break
            except Exception:
                continue

    pairs = _select_candidate_pairs(items, idxs, embs)
    if not pairs:
        return {
            "available": False,
            "mean01": None,
            "n_pairs": 0,
            "label_counts": {},
            "pairs": [],
            "fallback_used": "none",
        }

    results: List[PairJudgeResult] = []
    label_counts: DefaultDict[str, int] = defaultdict(int)

    for a, b in pairs:
        pa = items[a].get("image_path") or items[a].get("local_image")
        pb = items[b].get("image_path") or items[b].get("local_image")
        if not (isinstance(pa, str) and os.path.exists(pa) and isinstance(pb, str) and os.path.exists(pb)):
            continue

        sim01 = None
        if embs and cosine_similarity01 is not None and (a in embs) and (b in embs):
            try:
                sim01 = float(cosine_similarity01(embs[a], embs[b]))
            except Exception:
                sim01 = None

        ctx_a = _pack_context(items[a])
        ctx_b = _pack_context(items[b])

        r = _llm_pair_judge(pa, pb, ctx_a, ctx_b, sim01)
        r.pair = (a, b)
        results.append(r)
        label_counts[r.relation] += 1

    if not results:
        return {
            "available": False,
            "mean01": None,
            "n_pairs": 0,
            "label_counts": {},
            "pairs": [],
            "fallback_used": "none",
        }

    # weighted average by confidence; if all conf==0, fall back to plain mean
    num = 0.0
    den = 0.0
    plain = []
    for r in results:
        s = _clamp01(r.consistency_score, default=0.5)
        w = _clamp01(r.confidence, default=0.0)
        plain.append(s)
        num += s * w
        den += w
    mean01 = (num / den) if den > 1e-9 else (sum(plain) / max(1, len(plain)))

    return {
        "available": True,
        "mean01": float(_clamp01(mean01, default=0.5)),
        "n_pairs": int(len(results)),
        "label_counts": dict(label_counts),
        "pairs": [
            {
                "a": int(r.pair[0]),
                "b": int(r.pair[1]),
                "sim01": r.sim01,
                "relation": r.relation,
                "score": float(_clamp01(r.consistency_score, default=0.5)),
                "confidence": float(_clamp01(r.confidence, default=0.0)),
                "notes": r.notes,
            }
            for r in results
        ],
        "fallback_used": "llm" if MM.img_img_use_llm_judge else "none",
    }
