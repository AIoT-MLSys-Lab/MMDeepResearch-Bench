from __future__ import annotations

import re
from typing import Dict, List

from .scoring_math import safe_div

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
# Support http(s) and file:// URLs
URL_RE = re.compile(r"(?:https?|file)://[^\s\]\)]+", re.IGNORECASE)
# Allow 9xx local image citations (e.g., [901])
CIT_RE = re.compile(r"\[(\d{1,4})\]")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:https?|file)://[^)]+)\)", re.IGNORECASE)


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text or "")


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[。！？!?\.])\s+|\n{2,}", text or "")
    return [c.strip() for c in chunks if c and c.strip()]


def extract_markdown_headings(text: str) -> List[str]:
    hs = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if m:
            hs.append(m.group(1).strip())
    return hs


def count_paragraphs(text: str) -> int:
    paras = [p.strip() for p in re.split(r"\n{2,}", text or "") if p.strip()]
    return len(paras)


def ttr(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    forms = {w.lower() for w in words}
    return safe_div(len(forms), len(words), 0.0)


def title_density(text: str) -> float:
    headings = extract_markdown_headings(text)
    paras = count_paragraphs(text)
    return safe_div(len(headings), max(1, paras), 0.0)


def long_sentence_ratio(text: str) -> float:
    sents = split_sentences(text)
    if not sents:
        return 0.0
    long_cnt = 0
    for s in sents:
        words = tokenize_words(s)
        if len(s) > 35 or len(words) > 25:
            long_cnt += 1
    return safe_div(long_cnt, len(sents), 0.0)


def looks_like_image_url(url: str) -> bool:
    u = (url or "").lower()
    # file:///.../x.jpg also works
    u2 = u.split("?", 1)[0]
    return any(u2.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff"])


def normalize_report_type(report_type: str) -> str:
    rt = (report_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if rt.startswith("research_") or rt in {"research", "academic", "publishable", "top_conference", "deep"}:
        return "research"
    if "daily_easy" in rt or rt in {"easy"}:
        return "daily_easy"
    if "daily_hard" in rt or rt in {"hard"}:
        return "daily_hard"
    if "daily" in rt:
        return "daily_medium"
    return "daily_medium"


def find_references_block(report_text: str) -> str:
    lower = (report_text or "").lower()
    candidates = ["references", "reference", "参考文献", "参考资料", "引用", "來源", "来源"]
    idx = -1
    for c in candidates:
        j = lower.rfind(c.lower())
        if j > idx:
            idx = j
    return report_text[idx:] if idx != -1 else (report_text or "")


def parse_references(report_text: str) -> Dict[str, str]:
    tail = find_references_block(report_text)
    ref_map: Dict[str, str] = {}
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue

        # [1] https://...  |  [901] file://...
        m = re.match(r"^\[?(\d{1,4})\]?\s*[:\.\-]?\s*((?:https?|file)://\S+)\s*$", line, re.IGNORECASE)
        if m:
            ref_map[m.group(1)] = m.group(2).strip()
            continue

        # - [1] https://...
        m2 = re.match(r"^[\-\*]\s*\[?(\d{1,4})\]?\s*((?:https?|file)://\S+)\s*$", line, re.IGNORECASE)
        if m2:
            ref_map[m2.group(1)] = m2.group(2).strip()
            continue

    return ref_map
