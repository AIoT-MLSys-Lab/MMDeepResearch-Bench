from __future__ import annotations

import urllib.parse


def domain_of(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        if (p.scheme or "").lower() == "file":
            return "local"
        d = (p.netloc or "").lower()
        return d if d else ""
    except Exception:
        return ""


def source_is_prohibited(domain: str) -> bool:
    d = (domain or "").lower()
    if not d or d == "local":
        return False
    prohibited = [
        "reddit.com", "quora.com", "zhihu.com", "weibo.com",
        "x.com", "twitter.com", "facebook.com", "tiktok.com", "instagram.com",
        "medium.com", "blogspot.", "substack.com", "stackexchange.com",
    ]
    return any(x in d for x in prohibited)


def source_is_authoritative(domain: str) -> bool:
    d = (domain or "").lower()
    if not d or d == "local":
        return False
    if d.endswith(".gov") or d.endswith(".edu") or d.endswith(".int"):
        return True
    if any(k in d for k in ["who.int", "un.org", "worldbank.org", "imf.org", "oecd.org"]):
        return True
    return False


def classify_source_category(domain: str) -> str:
    if not domain:
        return "unknown"
    if domain == "local":
        # Dataset-local files are allowed but not "authoritative web sources"
        return "acceptable"
    if source_is_prohibited(domain):
        return "prohibited"
    if source_is_authoritative(domain):
        return "authoritative"
    return "acceptable"
