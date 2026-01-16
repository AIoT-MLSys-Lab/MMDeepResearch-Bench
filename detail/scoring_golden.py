from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .scoring_math import safe_div
from .scoring_text import (
    URL_RE, CIT_RE, MD_LINK_RE,
    split_sentences, parse_references, looks_like_image_url
)
from .scoring_web import is_image_dependent_claim


def estimate_relevance_weight(claim: str) -> float:
    c = claim or ""
    w = 1.0
    if len(c) >= 120:
        w += 0.3
    if re.search(r"\d", c):
        w += 0.2
    if any(k in c.lower() for k in ["because", "therefore", "causes", "due to", "impact", "trend", "increase", "decrease", "compare", "contrast"]):
        w += 0.2
    if any(k in c for k in ["因此", "所以", "导致", "影响", "趋势", "上升", "下降", "对比", "比较", "原因"]):
        w += 0.2
    return float(w)


def is_key_claim(claim: str) -> bool:
    c = claim or ""
    if len(c) >= 50:
        return True
    if re.search(r"\d", c) and len(c) >= 25:
        return True
    if any(k in c.lower() for k in ["key finding", "conclusion", "recommend", "main reason"]):
        return True
    if any(k in c for k in ["结论", "建议", "关键", "主要原因", "核心发现"]):
        return True
    return False


def _pairs_from_citations(report_text: str, max_pairs: int) -> List[Dict[str, Any]]:
    """Extract (claim, url) pairs from sentence-level inline citations.

    Enhancement for image-image consistency:
      - Keep sentence-level citation groups:
          - sentence_citation_ids: list of citation ids in this sentence
          - sentence_urls: list of resolved URLs in this sentence
        so downstream MM can detect sentences that cite multiple images and build
        image-image comparison items.
    """
    ref_map = parse_references(report_text)
    pairs: List[Dict[str, Any]] = []

    for s in split_sentences(report_text):
        cids = CIT_RE.findall(s)
        if not cids:
            continue

        sent_urls: List[str] = []
        for cid in cids:
            u = ref_map.get(cid)
            if u and u not in sent_urls:
                sent_urls.append(u)

        claim = CIT_RE.sub("", s).strip()
        claim = re.sub(r"\s+", " ", claim)
        if not claim:
            continue

        uniq_cids = list(dict.fromkeys(cids))
        for cid in uniq_cids:
            url = ref_map.get(cid)
            if not url:
                continue
            pairs.append(
                {
                    "claim": claim,
                    "url": url,
                    "citation_id": cid,
                    "sentence_citation_ids": uniq_cids,
                    "sentence_urls": sent_urls,
                }
            )
            if len(pairs) >= max_pairs:
                return pairs

    return pairs

def _pairs_from_markdown_links(report_text: str, max_pairs: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for s in split_sentences(report_text):
        for m in MD_LINK_RE.finditer(s):
            url = (m.group(2) or "").strip()
            if not url:
                continue
            claim = MD_LINK_RE.sub("", s).strip()
            claim = re.sub(r"\s+", " ", claim) or m.group(1).strip()
            pairs.append({"claim": claim, "url": url, "citation_id": None})
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _pairs_from_raw_urls(report_text: str, max_pairs: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for line in (report_text or "").splitlines():
        urls = URL_RE.findall(line)
        if not urls:
            continue
        claim = URL_RE.sub("", line).strip(" \t-")
        claim = re.sub(r"\s+", " ", claim)
        if not claim:
            claim = "Evidence referenced in report."
        for u in urls:
            pairs.append({"claim": claim, "url": u, "citation_id": None})
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def extract_claim_pairs(report_text: str, max_pairs: int = 40) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (raw_pairs, enriched_pairs).
    enriched_pairs include:
      - relevance_weight
      - is_key_claim
      - depends_on_image
    """
    raw = _pairs_from_citations(report_text, max_pairs=max_pairs)
    if len(raw) < max_pairs:
        raw += _pairs_from_markdown_links(report_text, max_pairs=max_pairs - len(raw))
    if len(raw) < max_pairs:
        raw += _pairs_from_raw_urls(report_text, max_pairs=max_pairs - len(raw))

    seen = set()
    enriched: List[Dict[str, Any]] = []
    for p in raw:
        claim = (p.get("claim") or "").strip()
        url = (p.get("url") or "").strip()
        if not claim or not url:
            continue
        key = (claim[:180], url)
        if key in seen:
            continue
        seen.add(key)

        enriched.append({
            "claim": claim,
            "url": url,
            "citation_id": p.get("citation_id"),
            "relevance_weight": estimate_relevance_weight(claim),
            "is_key_claim": is_key_claim(claim),
            "depends_on_image": bool(is_image_dependent_claim(claim) or looks_like_image_url(url)),
        })

        if len(enriched) >= max_pairs:
            break

    return raw, enriched
