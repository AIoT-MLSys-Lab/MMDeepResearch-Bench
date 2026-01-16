# pachong.py
# 更稳的“网页 -> 图片”解析与下载：
# - requests.Session 保留 cookies
# - headers + referer
# - 优先 og:image / itemprop=image / link rel=image_src / twitter:image
# - 支持 img/picture/source 的 srcset / data-srcset，选最大分辨率
# - 懒加载字段：data-src/data-original/data-lazy-src 等
# - 下载失败会输出 status / content-type 便于排查
# - 新增：遇到 401/403（反爬/防盗链）自动降级重试（origin referer / 无 referer）

from __future__ import annotations

def _file_url_to_path(url: str) -> str:
    """Convert file:// URL to local path (Windows-friendly)."""
    try:
        u = urlparse(url)
        p = unquote(u.path or "")
        if re.match(r"^/[A-Za-z]:/", p):
            p = p[1:]
        return p
    except Exception:
        return url

def _is_file_url(s: str) -> bool:
    return (s or "").lower().startswith("file://")

import os
import re
import math
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import shutil
from urllib.parse import urlparse, unquote

IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg")

BAD_KEYWORDS = [
    "logo", "icon", "favicon", "avatar", "sprite", "placeholder",
    "default", "folder", "loading", "blank", "spacer", "ads", "tracker"
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

IMAGE_HEADERS = {
    **DEFAULT_HEADERS,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def is_url(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")


def jina_proxy(url: str) -> str:
    # r.jina.ai 支持：https://r.jina.ai/http(s)://xxx
    if url.startswith("http://"):
        return "https://r.jina.ai/http://" + url[len("http://"):]
    if url.startswith("https://"):
        return "https://r.jina.ai/https://" + url[len("https://"):]
    return url


def safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = "".join(c for c in name if c not in r'\/:*?"<>|')
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:80] if name else "image"


def looks_like_bad_image_url(url: str) -> bool:
    u = (url or "").lower()
    if any(k in u for k in BAD_KEYWORDS):
        return True
    return False


def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    ext = (ext or "").lower()
    if ext in IMG_EXTS:
        return ext
    return ".jpg"


def guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    if "bmp" in ct:
        return ".bmp"
    if "svg" in ct:
        return ".svg"
    if "tiff" in ct:
        return ".tiff"
    return ".jpg"


def simple_tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text or "")]


def jaccard_sim(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def parse_srcset(srcset: str, base_url: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if not srcset:
        return out
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    for p in parts:
        segs = p.split()
        if not segs:
            continue
        u = urljoin(base_url, segs[0].strip())
        w = 0
        if len(segs) >= 2:
            m = re.match(r"(\d+)(w|x)", segs[1].strip())
            if m:
                try:
                    w = int(m.group(1))
                except Exception:
                    w = 0
        out.append((u, w))
    return out


def pick_best_from_srcset(srcset: str, base_url: str) -> tuple[str, int] | None:
    cands = parse_srcset(srcset, base_url)
    if not cands:
        return None
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0]


class PageFetcher:
    def __init__(self):
        self.session = requests.Session()

    def get_html(self, url: str, referer: str | None = None, use_jina: bool = False, timeout: int = 18) -> tuple[str, str]:
        headers = dict(DEFAULT_HEADERS)
        if referer:
            headers["Referer"] = referer
        target = jina_proxy(url) if use_jina else url
        resp = self.session.get(target, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return (resp.text or ""), str(resp.url)

    def download_image(self, url: str, filepath: str, referer: str | None = None, timeout: int = 18, min_bytes: int = 1024) -> tuple[bool, str]:
        """
        下载图片：
        - 默认带 Referer（可提升防盗链成功率）
        - 若遇到 401/403（反爬/防盗链），会自动再试：
          1) 使用站点 origin 作为 Referer
          2) 不带 Referer
        """
        def _attempt(_referer: str | None) -> tuple[bool, str, str, int]:
            headers = dict(IMAGE_HEADERS)
            if _referer:
                headers["Referer"] = _referer
            resp = self.session.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            ct = resp.headers.get("Content-Type", "")
            status = int(getattr(resp, "status_code", 0) or 0)
            if status >= 400:
                return False, f"HTTP {status} {ct}", ct, status
            data = resp.content
            if not data or len(data) < min_bytes:
                return False, f"too_small {0 if not data else len(data)} bytes ({ct})", ct, status

            ext = os.path.splitext(filepath)[1].lower()
            if (not ext) or (ext not in IMG_EXTS):
                filepath2 = os.path.splitext(filepath)[0] + guess_ext_from_content_type(ct)
            else:
                filepath2 = filepath

            with open(filepath2, "wb") as f:
                f.write(data)

            if "text/html" in (ct or "").lower():
                try:
                    os.remove(filepath2)
                except Exception:
                    pass
                return False, f"not_image_content_type ({ct})", ct, status

            return True, f"ok {len(data)} bytes ({ct})", ct, status

        try:
            ok, why, ct, status = _attempt(referer)
            if ok:
                return True, why

            if status in (401, 403):
                try:
                    u = urlparse(url)
                    origin = f"{u.scheme}://{u.netloc}/" if u.scheme and u.netloc else None
                except Exception:
                    origin = None

                if origin and origin != referer:
                    ok2, why2, _, _ = _attempt(origin)
                    if ok2:
                        return True, why2

                ok3, why3, _, _ = _attempt(None)
                if ok3:
                    return True, why3

                return False, f"blocked {why}"

            return False, why

        except Exception as e:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
            return False, f"exception {e}"


def extract_image_candidates(html: str, base_url: str, caption: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cap_words = simple_tokenize(caption)

    candidates: list[dict] = []

    meta_props = [
        ("meta", {"property": "og:image"}, "content"),
        ("meta", {"name": "twitter:image"}, "content"),
        ("meta", {"property": "og:image:url"}, "content"),
        ("meta", {"itemprop": "image"}, "content"),
        ("link", {"rel": "image_src"}, "href"),
    ]
    for tag_name, attrs, key in meta_props:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get(key):
            u = urljoin(base_url, tag.get(key).strip())
            if is_url(u) and not looks_like_bad_image_url(u):
                candidates.append({"url": u, "score": 1.00, "why": "meta", "pos": 0, "w": 9999})

    pos = 0
    for src in soup.find_all("source"):
        pos += 1
        ss = src.get("srcset") or src.get("data-srcset") or ""
        best = pick_best_from_srcset(ss, base_url)
        if best:
            u, w = best
            if is_url(u) and not looks_like_bad_image_url(u):
                candidates.append({"url": u, "score": 0.85, "why": "source_srcset", "pos": pos, "w": w})

    pos = 0
    for img in soup.find_all("img"):
        pos += 1

        alt = ((img.get("alt") or "") + " " + (img.get("title") or "")).strip()
        alt_words = simple_tokenize(alt)
        sim = jaccard_sim(cap_words, alt_words)

        ss = img.get("srcset") or img.get("data-srcset") or ""
        best = pick_best_from_srcset(ss, base_url)
        if best:
            u, w = best
            if is_url(u) and not looks_like_bad_image_url(u):
                score = 0.60 + 0.35 * sim + min(0.05, math.log(max(2, w), 10) / 50)
                candidates.append({"url": u, "score": score, "why": "img_srcset", "pos": pos, "w": w})

        u = (
            img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy-src")
            or img.get("data-url")
            or img.get("src")
            or ""
        ).strip()

        if not u:
            continue

        u = urljoin(base_url, u)
        if not is_url(u) or looks_like_bad_image_url(u):
            continue

        score = 0.50 + 0.35 * sim
        candidates.append({"url": u, "score": score, "why": "img", "pos": pos, "w": 0})

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = urljoin(base_url, href)
        if not is_url(u):
            continue
        p = urlparse(u).path.lower()
        if any(p.endswith(ext) for ext in IMG_EXTS) and not looks_like_bad_image_url(u):
            score = 0.30
            candidates.append({"url": u, "score": score, "why": "a_img", "pos": 9999, "w": 0})

    uniq = {}
    for c in candidates:
        u = c["url"]
        if u not in uniq or c["score"] > uniq[u]["score"]:
            uniq[u] = c
    candidates = list(uniq.values())

    candidates.sort(key=lambda x: (x.get("score", 0.0), x.get("w", 0)), reverse=True)
    return candidates[:18]


def resolve_best_image_for_item(
    item: dict,
    out_dir: str = "images",
    max_try: int = 5,
) -> dict:
    """Resolve an item's image_path to a local image file.

    Supports:
      - direct image URLs
      - web pages (extract og:image etc.)
      - local paths
      - file:// URLs (local references)
    """
    fetcher = PageFetcher()

    image_path = (item.get("image_path") or "").strip()
    caption = (item.get("caption") or "Evidence referenced in report.").strip()
    idx = int(item.get("index") or 0)

    os.makedirs(out_dir, exist_ok=True)
    referer = item.get("source_url") or None

    # ---- Local file path / file:// URL ----
    if _is_file_url(image_path):
        local = _file_url_to_path(image_path)
        if local and os.path.exists(local):
            ext = os.path.splitext(local)[1] or ".jpg"
            fp = os.path.join(out_dir, f"{idx:02d}_{safe_filename(caption or f'image_{idx}')}_local{ext}")
            try:
                shutil.copy2(local, fp)
                item["missing_image"] = False
                item["local_image"] = fp
                item["picked_image_url"] = image_path
                item["picked_image_why"] = "file_url"
                item["picked_image_score"] = 1.0
                item["image_path"] = fp
                return item
            except Exception as e:
                item["missing_image"] = True
                item["local_image"] = None
                item["download_error"] = f"file_copy_failed: {e}"
                return item
        item["missing_image"] = True
        item["local_image"] = None
        item["download_error"] = "file_url_not_found"
        return item

    if image_path and os.path.exists(image_path):
        ext = os.path.splitext(image_path)[1] or ".jpg"
        fp = os.path.join(out_dir, f"{idx:02d}_{safe_filename(caption or f'image_{idx}')}_local{ext}")
        try:
            shutil.copy2(image_path, fp)
            item["missing_image"] = False
            item["local_image"] = fp
            item["picked_image_url"] = image_path
            item["picked_image_why"] = "local_path"
            item["picked_image_score"] = 1.0
            item["image_path"] = fp
            return item
        except Exception as e:
            item["missing_image"] = True
            item["local_image"] = None
            item["download_error"] = f"local_copy_failed: {e}"
            return item

    # ---- Direct image URL ----
    if is_url(image_path) and any(image_path.lower().split("?", 1)[0].endswith(ext) for ext in IMG_EXTS):
        ext = guess_ext_from_url(image_path)
        fp = os.path.join(out_dir, f"{idx:02d}_{safe_filename(caption or f'image_{idx}')}_1{ext}")
        ok, why = fetcher.download_image(image_path, fp, referer=referer)
        if ok:
            item["missing_image"] = False
            item["local_image"] = fp
            item["picked_image_url"] = image_path
            item["picked_image_why"] = "direct_url"
            item["picked_image_score"] = 1.0
            item["image_path"] = fp
        else:
            item["missing_image"] = True
            item["local_image"] = None
            item["download_error"] = why
        return item

    # ---- Web page URL: fetch HTML and extract candidate images ----
    html = ""
    final_url = image_path
    for use_jina in [False, True]:
        try:
            html, final_url = fetcher.get_html(image_path, referer=referer, use_jina=use_jina)
            if html:
                break
        except Exception as e:
            item["page_fetch_error"] = str(e)

    if not html:
        item["missing_image"] = True
        item["local_image"] = None
        item["download_error"] = "page_fetch_failed"
        return item

    cands = extract_image_candidates(html, base_url=final_url, caption=caption)
    tried = 0
    for c in cands[:max_try]:
        tried += 1
        u = c.get("url")
        if not u:
            continue
        ext = guess_ext_from_url(u)
        fp = os.path.join(out_dir, f"{idx:02d}_{safe_filename(caption or f'image_{idx}')}_{tried}{ext}")

        ok, why = fetcher.download_image(u, fp, referer=final_url)
        if ok:
            item["missing_image"] = False
            item["local_image"] = fp
            item["picked_image_url"] = u
            item["picked_image_why"] = c.get("why")
            item["picked_image_score"] = round(float(c.get("score", 0.0)), 4)
            item["num_candidates"] = len(cands)
            item["image_path"] = fp
            return item

        item["last_download_error"] = why

    item["missing_image"] = True
    item["local_image"] = None
    item["download_error"] = item.get("last_download_error", "no_image_candidate_downloaded")
    item["num_candidates"] = len(cands)
    return item

def resolve_images_from_items(items: list[dict], out_dir: str = "images") -> list[dict]:
    """Resolve images for all items.

    For img_txt items: resolve_best_image_for_item

    For img_img items (pair): resolve BOTH images into local files:
      - image_path -> local A (and overwritten to that local file)
      - image_path_b -> local B
      - local_images -> [A, B]
    """
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for it in items:
        if (it.get("mm_mode") == "img_img") and it.get("image_path_b"):
            a_seed = (it.get("image_path") or "").strip()
            b_seed = (it.get("image_path_b") or "").strip()

            it_a = {"index": it.get("index"), "image_path": a_seed, "caption": (it.get("caption") or "pair") + "_A", "source_url": it.get("source_url")}
            it_b = {"index": it.get("index"), "image_path": b_seed, "caption": (it.get("caption") or "pair") + "_B", "source_url": it.get("source_url")}

            it_a = resolve_best_image_for_item(it_a, out_dir=out_dir)
            it_b = resolve_best_image_for_item(it_b, out_dir=out_dir)

            a_local = it_a.get("image_path")
            b_local = it_b.get("image_path")

            it["local_images"] = [p for p in [a_local, b_local] if p]
            it["local_image"] = a_local
            it["missing_image"] = not (a_local and b_local)

            if a_local:
                it["image_path"] = a_local
            if b_local:
                it["image_path_b"] = b_local

            out.append(it)
        else:
            out.append(resolve_best_image_for_item(it, out_dir=out_dir))
    return out

