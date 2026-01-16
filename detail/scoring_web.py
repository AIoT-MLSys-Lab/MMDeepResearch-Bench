from __future__ import annotations

import re
import urllib.request
import urllib.parse
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import CRAWL
from .scoring_math import clamp01, safe_div
from .scoring_text import tokenize_words, looks_like_image_url
from .scoring_sources import domain_of

FETCH_CACHE: Dict[str, Tuple[str, str]] = {}

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")


def fetch_url(url: str) -> Tuple[str, str]:
    if url in FETCH_CACHE:
        return FETCH_CACHE[url]

    req = urllib.request.Request(
        url,
        headers={"User-Agent": getattr(CRAWL, "user_agent", "Mozilla/5.0")},
    )
    try:
        with urllib.request.urlopen(req, timeout=getattr(CRAWL, "timeout_s", 15)) as resp:
            ct = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(getattr(CRAWL, "max_bytes", 2_000_000))

            enc = "utf-8"
            m = re.search(r"charset=([A-Za-z0-9_\-]+)", ct)
            if m:
                enc = m.group(1)

            try:
                s = raw.decode(enc, errors="ignore")
            except Exception:
                s = raw.decode("utf-8", errors="ignore")

            FETCH_CACHE[url] = (ct, s)
            return ct, s
    except Exception:
        FETCH_CACHE[url] = ("", "")
        return "", ""


def html_to_text(html: str, max_chars: int) -> str:
    h = html or ""
    h = SCRIPT_STYLE_RE.sub(" ", h)
    h = re.sub(r"(?is)<br\s*/?>", "\n", h)
    h = re.sub(r"(?is)</p\s*>", "\n\n", h)
    h = TAG_RE.sub(" ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h[:max_chars]


def is_image_dependent_claim(claim: str) -> bool:
    c = (claim or "").lower()
    cues = [
        "figure", "fig.", "chart", "graph", "plot", "diagram", "table", "screenshot", "image",
        "as shown", "shown below", "see below",
        "图", "如下图", "见图", "截图", "屏摄", "图表", "曲线", "柱状", "折线",
    ]
    return any(k in c for k in cues)


def extract_img_candidates_with_context(html: str, max_scan_chars: int) -> List[Dict[str, Any]]:
    if not html:
        return []
    h = html[:max_scan_chars]
    out: List[Dict[str, Any]] = []

    for m in re.finditer(r"(?is)<img\b[^>]*>", h):
        tag = m.group(0)
        src_m = re.search(r'(?is)\bsrc\s*=\s*["\']([^"\']+)["\']', tag)
        if not src_m:
            continue
        src = src_m.group(1).strip()
        alt_m = re.search(r'(?is)\balt\s*=\s*["\']([^"\']+)["\']', tag)
        alt = alt_m.group(1).strip() if alt_m else ""
        pos = m.start()
        w0 = max(0, pos - 500)
        w1 = min(len(h), pos + 500)
        ctx = TAG_RE.sub(" ", h[w0:w1])
        ctx = re.sub(r"\s+", " ", ctx).strip()
        out.append({"src": src, "alt": alt, "pos": pos, "ctx": ctx})

    out.sort(key=lambda x: int(x.get("pos", 0)))
    for i, it in enumerate(out):
        it["rank"] = i + 1
    return out


def img_relevance_score(claim: str, alt: str, ctx: str) -> float:
    claim_words = {w.lower() for w in tokenize_words(claim)}
    bag_words = {w.lower() for w in tokenize_words((alt or "") + " " + (ctx or ""))}
    if not claim_words:
        return 0.0
    inter = len(claim_words & bag_words)
    return safe_div(inter, len(claim_words), 0.0)


def depth_score_from_rank(rank: int) -> float:
    ideal = int(getattr(CRAWL, "ideal_img_rank", 3))
    maxr = int(getattr(CRAWL, "max_img_rank", 25))
    if rank <= ideal:
        return 1.0
    if rank >= maxr:
        return 0.10
    t = safe_div(rank - ideal, maxr - ideal, 1.0)
    return clamp01(1.0 - 0.90 * t)


def fetch_evidence_excerpt_with_patience(url: str, claim: str, patience_chars: int) -> Dict[str, Any]:
    """
    Returns:
      - excerpt_text: truncated by patience budget
      - depth_score: gradient decay for image-dependent claims on webpages
      - meta: content_type, domain, image_found, best_image (alt/src/rank)

    Special handling:
      - Direct image URLs / image content-type are returned as a structured stub excerpt (DIRECT_IMAGE_FILE)
        so the text judge stays conservative and downstream MM can validate pixels.
    """
    # Local file:// evidence (typically dataset images). Treat as DIRECT_IMAGE_FILE for text judge.
    if (url or "").lower().startswith("file://"):
        try:
            pu = urllib.parse.urlparse(url)
            p = urllib.parse.unquote(pu.path or "")
            # Windows file URIs often look like /C:/...
            if re.match(r"^/[A-Za-z]:/", p):
                p = p[1:]
            fp = Path(p)
            ct = mimetypes.guess_type(str(fp))[0] or ""
            meta: Dict[str, Any] = {
                "content_type": ct,
                "domain": "local",
                "image_found": True,
                "best_image": {"src": url, "alt": "", "rank": 1},
                "fetch_ok": fp.exists(),
                "evidence_type": "DIRECT_IMAGE_FILE",
            }
            excerpt = (
                "EVIDENCE TYPE: DIRECT_IMAGE_FILE\n"
                f"IMAGE_URL: {url}\n"
                f"FETCH_OK: {meta['fetch_ok']}\n"
                "NOTE: Local file evidence is not parsed in the text pipeline; defer pixel validation to MM stage.\n"
            )
            return {"excerpt_text": excerpt[:8000], "depth_score": 1.0, "meta": meta}
        except Exception:
            meta = {
                "content_type": "",
                "domain": "local",
                "image_found": True,
                "best_image": {"src": url, "alt": "", "rank": 1},
                "fetch_ok": False,
                "evidence_type": "DIRECT_IMAGE_FILE",
            }
            excerpt = (
                "EVIDENCE TYPE: DIRECT_IMAGE_FILE\n"
                f"IMAGE_URL: {url}\n"
                "FETCH_OK: False\n"
                "NOTE: Local file evidence is not parsed in the text pipeline; defer pixel validation to MM stage.\n"
            )
            return {"excerpt_text": excerpt[:8000], "depth_score": 1.0, "meta": meta}

    ct, body = fetch_url(url)
    domain = domain_of(url)
    meta: Dict[str, Any] = {
        "content_type": ct,
        "domain": domain,
        "image_found": False,
        "best_image": None,
        "fetch_ok": bool(body),
    }

    # IMPORTANT: still emit DIRECT_IMAGE_FILE even if fetch failed.
    if looks_like_image_url(url) or (ct.startswith("image/") if ct else False):
        meta["image_found"] = True
        meta["best_image"] = {"src": url, "alt": "", "rank": 1}
        meta["evidence_type"] = "DIRECT_IMAGE_FILE"
        excerpt = (
            "EVIDENCE TYPE: DIRECT_IMAGE_FILE\n"
            f"IMAGE_URL: {url}\n"
            f"FETCH_OK: {meta['fetch_ok']}\n"
            "NOTE: Image pixels are not parsed in this text pipeline; defer pixel validation to MM stage.\n"
        )
        return {"excerpt_text": excerpt[:8000], "depth_score": 1.0, "meta": meta}

    if not body:
        return {"excerpt_text": "", "depth_score": 1.0, "meta": meta}

    is_html = ("text/html" in ct) or ("<html" in body.lower())
    if not is_html:
        return {"excerpt_text": body[: min(len(body), 40_000)], "depth_score": 1.0, "meta": meta}

    excerpt_text = html_to_text(body, max_chars=min(40_000, patience_chars))

    depth_score = 1.0
    if is_image_dependent_claim(claim) and not looks_like_image_url(url):
        imgs = extract_img_candidates_with_context(body, max_scan_chars=min(len(body), patience_chars))
        if imgs:
            best, best_rel = None, -1.0
            for it in imgs[: max(50, len(imgs))]:
                rel = img_relevance_score(claim, it.get("alt", ""), it.get("ctx", ""))
                if rel > best_rel:
                    best_rel = rel
                    best = it

            meta["image_found"] = True
            if best is not None:
                meta["best_image"] = {
                    "src": best.get("src", ""),
                    "alt": best.get("alt", ""),
                    "rank": int(best.get("rank", 999)),
                }
                if best_rel < 0.05:
                    depth_score = 0.45
                else:
                    depth_score = depth_score_from_rank(int(best.get("rank", 999)))

                excerpt_text = (
                    excerpt_text[:6000]
                    + "\n\nIMAGE_CONTEXT_HINT:\n"
                    + f"- img_rank: {meta['best_image']['rank']}\n"
                    + f"- img_alt: {meta['best_image']['alt']}\n"
                    + f"- img_src: {meta['best_image']['src']}\n"
                    + f"- local_text: {best.get('ctx','')[:1200]}\n"
                )
            else:
                depth_score = 0.40
        else:
            depth_score = 0.30

    return {"excerpt_text": excerpt_text, "depth_score": float(depth_score), "meta": meta}
