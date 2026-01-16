# evidence_utils.py
"""
Utilities for extracting evidence maps ("golden.json") from a deep research report.

Key idea:
- For each generated deep report, we automatically build a self-supervised
  evidence map from the report itself.
- This evidence map is later consumed by the Evidence LLM judge to score
  faithfulness and reasoning fidelity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


URL_PATTERN = re.compile(r"https?://[^\s\)\]\}<>\"']+")


@dataclass
class EvidenceClaim:
    """Single atomic, checkable claim aligned to at least one URL."""
    id: str
    statement: str
    link: str


def _split_into_sentences(text: str) -> List[str]:
    """
    Very lightweight sentence splitter, only used to get a cleaner 'statement'
    around a URL. This does not need to be perfect; the LLM judge will refine.
    """
    # Normalize newlines and collapse excessive whitespace
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Split on period/question/exclamation marks while keeping them attached
    # This is intentionally simple.
    parts = re.split(r"(?<=[。！？!?\.])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _extract_claims_from_line(line: str, next_claim_id: int) -> List[EvidenceClaim]:
    """
    Given a line of markdown, find all URLs and map them to local statements.
    """
    urls = URL_PATTERN.findall(line)
    if not urls:
        return []

    # Roughly get the sentence(s) that contain this line
    sentences = _split_into_sentences(line)
    if not sentences:
        sentences = [line.strip()]

    claims: List[EvidenceClaim] = []
    # Use the whole sentence as the statement for now; the LLM judge will refine.
    # We associate each URL with the closest sentence in the line.
    for url in urls:
        # Choose the first sentence that actually contains the URL or some part of it
        stmt = None
        for s in sentences:
            if url in s:
                stmt = s
                break
        if stmt is None:
            # fallback: use the whole line
            stmt = line.strip()

        cid = f"{next_claim_id}"
        claims.append(EvidenceClaim(id=cid, statement=stmt, link=url))
        next_claim_id += 1

    return claims


def build_evidence_map_from_report(
    report_markdown: str,
    question_id: str,
    golden_path: Optional[Path] = None,
) -> Dict:
    """
    Auto-construct a self-supervised evidence map ("golden") from a single
    deep research report.

    - Scans the markdown for all URLs.
    - For each URL, extracts a local statement (sentence) as a checkable claim.
    - Returns a dict:
        {
          "question_id": "...",
          "claims": [
            {"id": "1", "statement": "...", "link": "https://..."},
            ...
          ]
        }
    - If golden_path is provided, overwrites that file on disk.

    This golden is *not* an external ground truth; it is an evidence map
    that the objective LLM judge will later refine and critique for
    faithfulness / reasoning fidelity.
    """
    lines = report_markdown.splitlines()
    claims: List[EvidenceClaim] = []
    seen_pairs = set()
    next_id = 1

    for line in lines:
        if "http://" not in line and "https://" not in line:
            continue
        line_claims = _extract_claims_from_line(line, next_id)
        for c in line_claims:
            key = (c.statement, c.link)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            claims.append(c)
            next_id = int(c.id) + 1

    golden = {
        "question_id": question_id,
        "claims": [
            {"id": c.id, "statement": c.statement, "link": c.link} for c in claims
        ],
    }

    if golden_path is not None:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with golden_path.open("w", encoding="utf-8") as f:
            json.dump(golden, f, ensure_ascii=False, indent=2)

    return golden
