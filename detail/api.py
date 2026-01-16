# api.py
"""
统一封装 Gemini 调用的工具函数(显式传 api_key 版本,无命令行参数)

新增：Azure OpenAI (OpenAI Python SDK) 的轻量封装，用于切换被测模型。

依赖:
    pip install google-genai
    pip install openai

主要导出函数:
    - set_gemini_api_key(key: str)
    - get_gemini_client(api_key=None)
    - gemini_text(...)
    - gemini_chat(...) -> 用于报告生成
    - gemini_json(...) -> 用于评分 Judge
    - gemini_mm_qa(...)

Azure 新增导出函数:
    - set_azure_openai_api_key(key: str)
    - set_azure_openai_base_url(base_url: str)
    - set_azure_openai_endpoint(endpoint: str)
    - set_azure_openai_api_version(version: str)
    - azure_chat(...)
    - azure_json(...)
    - llm_json(...)  # 统一 JSON 输出接口（Gemini/Azure）
"""

from __future__ import annotations

import os
import json
from typing import Any, Optional, List, Dict

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError as e:
    genai = None
    types = None
    APIError = None
    _IMPORT_ERR = e

# 默认主力模型
DEFAULT_MODEL = "gemini-2.5-flash"

# 全局缓存一个 key
_API_KEY: Optional[str] = None


# =========================
#   Gemini Client 初始化
# =========================

def set_gemini_api_key(key: str) -> None:
    """显式设置全局 Gemini API key。"""
    global _API_KEY
    _API_KEY = key

API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY: set_gemini_api_key(API_KEY)

def get_gemini_client(api_key: Optional[str] = None) -> "genai.Client":
    """返回一个已配置好的 Gemini client。"""
    if genai is None:
        raise ImportError(
            "未找到 google-genai 包,请先安装:\n"
            "    pip install google-genai\n"
        )

    key = api_key or _API_KEY or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Gemini API key 未配置。请调用 set_gemini_api_key('YOUR_KEY') 或设置环境变量 GEMINI_API_KEY。"
        )

    return genai.Client(api_key=key)


# =========================
#   Gemini: 结构化 JSON 输出 (For Scoring Judge)
# =========================

def gemini_json(
    prompt: str,
    json_schema: Dict[str, Any],
    model: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 81920,
    **extra_config: Any,
) -> str:
    """强制模型输出符合 json_schema 定义的 JSON 字符串。"""
    client = get_gemini_client(api_key)

    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=json_schema,
        **extra_config,
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[Gemini JSON Error] {e}")
        return "{}"


# =========================
#   Gemini: 多轮对话 Chat (For Report Generation)
# =========================

def gemini_chat(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 81920,
    **extra_config: Any,
) -> str:
    """
    Chat interface for report generation.

    Supports multimodal via:
      1) OpenAI-style content blocks in messages:
         - {"type":"text","text":...}
         - {"type":"image_url","image_url":{"url": "data:image/...;base64,..." | "http(s)://..." | "file://..." | "<local_path>"}}
      2) Legacy kwargs: image_paths (list[str] or str). These will be appended to the LAST user message.

    Important:
      - This function converts the above formats into Gemini `types.Content(parts=[...])` with inline image bytes.
      - So Gemini will ACTUALLY see the images (not just a text transcript).
    """
    client = get_gemini_client(api_key)

    import re
    import base64
    import mimetypes
    from pathlib import Path
    from urllib.parse import urlparse, unquote
    from urllib.request import urlopen, url2pathname

    # ----- helpers -----
    def _file_uri_to_path(uri: str) -> Path:
        parsed = urlparse(uri)
        p = url2pathname(unquote(parsed.path or ""))
        # Windows: /C:/... -> C:/...
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return Path(p)

    def _collapse_dup_image_dir(p: Path) -> Path:
        s = str(p).replace("\\", "/")
        low = s.lower()
        if "/image/image/" in low:
            parts = s.split("/")
            lowered = [x.lower() for x in parts]
            for i in range(len(parts) - 2):
                if lowered[i] == "image" and lowered[i + 1] == "image":
                    collapsed = "/".join(parts[: i + 1] + parts[i + 2 :])
                    p1 = Path(collapsed)
                    if p1.exists():
                        return p1
                    p2 = Path(collapsed.replace("/", "\\"))
                    if p2.exists():
                        return p2
                    break
        return p

    def _read_local_file_bytes(p: str) -> tuple[bytes, str]:
        p = (p or "").strip()
        if not p:
            raise FileNotFoundError("Empty path")

        if p.startswith("file://"):
            path = _file_uri_to_path(p)
        else:
            path = Path(p)

        try:
            if not path.is_absolute():
                path = path.resolve()
        except Exception:
            pass

        path = _collapse_dup_image_dir(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        return data, mime

    def _part_from_image_url(url: str) -> "types.Part":
        url = (url or "").strip()
        if not url:
            raise ValueError("Empty image url")

        # data url
        if url.startswith("data:image/"):
            m = re.match(r"^data:(image/[^;]+);base64,(.+)$", url, flags=re.DOTALL)
            if not m:
                raise ValueError("Invalid data:image base64 url")
            mime, b64data = m.group(1), m.group(2)
            b = base64.b64decode(b64data)
            return types.Part(inline_data=types.Blob(mime_type=mime, data=b))

        # http(s) url -> download
        if url.startswith("http://") or url.startswith("https://"):
            with urlopen(url) as r:
                b = r.read()
                mime = r.headers.get_content_type() if hasattr(r, "headers") else None
                if not mime or not str(mime).startswith("image/"):
                    mime = "image/jpeg"
                return types.Part(inline_data=types.Blob(mime_type=mime, data=b))

        # file:// or local path
        b, mime = _read_local_file_bytes(url)
        return types.Part(inline_data=types.Blob(mime_type=mime, data=b))

    def _ensure_text_part(parts: List["types.Part"]) -> None:
        # Gemini generally tolerates empty text, but keep at least one text part in user message.
        if not any(getattr(p, "text", None) is not None for p in parts):
            parts.insert(0, types.Part(text=""))

    # ----- read legacy image_paths (append to LAST user) -----
    image_paths = extra_config.pop("image_paths", None)
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if not isinstance(image_paths, list):
        image_paths = []
    image_paths = [str(p).strip() for p in image_paths if p and str(p).strip()]

    # ----- convert messages -> gemini_contents -----
    gemini_contents: List["types.Content"] = []
    system_instruction = None
    had_any_image = False

    # We may need to append legacy images to last user content
    # so first, copy messages shallowly
    msgs = list(messages or [])

    if image_paths:
        # find last user index
        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if (msgs[i].get("role") or "user").lower() == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            # if no user msg, create one
            msgs.append({"role": "user", "content": ""})
            last_user_idx = len(msgs) - 1

        c = msgs[last_user_idx].get("content", "")
        # normalize to blocks list
        if isinstance(c, list):
            blocks = c
        else:
            blocks = [{"type": "text", "text": str(c or "")}]

        # append as image_url blocks (local path or url); later will be converted to inline bytes
        for p in image_paths:
            blocks.append({"type": "image_url", "image_url": {"url": p}})
        msgs[last_user_idx]["content"] = blocks

    # build contents
    for msg in msgs:
        role_raw = (msg.get("role") or "user").strip().lower()
        content = msg.get("content", "")

        if role_raw == "system":
            # keep latest system instruction
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                # join text blocks
                ts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        ts.append(str(b.get("text", "")))
                system_instruction = "\n".join([t for t in ts if t]).strip()
            continue

        role = "user" if role_raw == "user" else "model"  # assistant/model
        parts: List["types.Part"] = []

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type")
                if t == "text":
                    parts.append(types.Part(text=str(block.get("text", "") or "")))
                elif t == "image_url":
                    u = (block.get("image_url") or {}).get("url")
                    if isinstance(u, str) and u.strip():
                        try:
                            parts.append(_part_from_image_url(u.strip()))
                            had_any_image = True
                        except Exception as e:
                            print(f"[Gemini Chat Warning] failed to parse image_url: {u} ({e})")
                else:
                    # ignore unknown block types
                    pass
        else:
            parts.append(types.Part(text=str(content or "")))

        if role == "user":
            _ensure_text_part(parts)

        gemini_contents.append(types.Content(role=role, parts=parts))

    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **extra_config,
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=cfg,
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[Gemini Chat Error] {e}")
        import traceback
        traceback.print_exc()
        return ""

def gemini_deep_research(
    messages: List[Dict[str, Any]],
    agent: str = "deep-research-pro-preview-12-2025",
    api_key: Optional[str] = None,
    # Polling (Deep Research is async/long-running)
    poll_interval: float = 10.0,
    max_wait_s: float = 60.0 * 60.0,  # 60 minutes
    # Agent config
    thinking_summaries: str = "none",  # "auto" | "none"
    # NOTE: DR agent requires background=True; store=True is recommended/required for background runs.
    background: bool = True,
    store: bool = True,
    stream: bool = False,
    raise_on_error: bool = False,
    **extra_config: Any,
) -> str:
    """
    Gemini Deep Research Agent wrapper (Interactions API).

    Why this exists:
      - The DR agent (e.g. deep-research-pro-preview-12-2025) is ONLY available via the Interactions API.
      - Using client.models.generate_content(...) will raise:
            400 INVALID_ARGUMENT: "This model only supports Interactions API."

    Input format:
      - Accepts the SAME 'messages' format as gemini_chat (OpenAI-style blocks incl. image_url),
        and converts them into Interactions API 'input' objects.
      - Supports images via:
          {"type":"image","data":"<BASE64>","mime_type":"image/png"}   (preferred)
        and will also accept http(s)/file/local paths, which are read and base64-encoded.

    Behavior:
      - Starts an interaction with background=True and polls until completed/failed (or timeout).
      - Returns the final text output (or "" on error if raise_on_error=False).
    """
    import time
    import re
    import base64
    import mimetypes
    from pathlib import Path
    from urllib.parse import urlparse, unquote
    from urllib.request import urlopen, url2pathname

    client = get_gemini_client(api_key)

    # NOTE: The Interactions API (and thus Deep Research) is only exposed in
    # google-genai >= 1.55.0. If the installed SDK is older, `client` won't have
    # a `.interactions` service. We fall back to direct REST calls in that case.
    import os
    import json as _json
    from urllib.request import Request as _Request
    from urllib.error import HTTPError as _HTTPError, URLError as _URLError

    _api_key_eff = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _rest_json(method: str, url: str, body: Optional[dict] = None) -> dict:
        if not _api_key_eff:
            raise ValueError("Gemini API key not found (set GEMINI_API_KEY / GOOGLE_API_KEY or pass api_key).")

        data = None
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": _api_key_eff,
        }
        if body is not None:
            data = _json.dumps(body).encode("utf-8")

        req = _Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req) as r:
                raw = r.read()
            return _json.loads(raw.decode("utf-8")) if raw else {}
        except _HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            raise RuntimeError(f"Interactions REST {method} failed: HTTP {e.code} {e.reason} {err_body}") from e
        except _URLError as e:
            raise RuntimeError(f"Interactions REST {method} failed: {e}") from e

    def _rest_create_and_poll() -> str:
        # Deep Research requires background=True (async); store=true is required with background.
        req_body = {
            "agent": agent,
            "input": input_items,
            "system_instruction": system_instruction,
            "background": bool(background),
            "store": True if background else bool(store),
            "stream": bool(stream),
            "agent_config": agent_cfg,
        }
        interaction = _rest_json("POST", "https://generativelanguage.googleapis.com/v1beta/interactions", req_body)
        interaction_id = interaction.get("id")
        if not interaction_id:
            # If API returned inline outputs without an id, try to extract text now.
            outs = interaction.get("outputs") or []
            for o in reversed(outs):
                if isinstance(o, dict) and o.get("type") == "text" and (o.get("text") or "").strip():
                    return o["text"]
            return ""

        start_t = time.time()
        while True:
            cur = _rest_json("GET", f"https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}")
            status = (cur.get("status") or "").lower()
            if status == "completed":
                outs = cur.get("outputs") or []
                for o in reversed(outs):
                    if isinstance(o, dict) and o.get("type") == "text" and (o.get("text") or "").strip():
                        return o["text"]
                return ""
            if status in ("failed", "cancelled", "canceled", "error"):
                raise RuntimeError(f"Deep Research interaction {interaction_id} {status}: {cur.get('error')}")
            if (time.time() - start_t) > float(max_wait_s):
                raise TimeoutError(f"Deep Research interaction timeout after {max_wait_s}s (id={interaction_id})")
            time.sleep(float(poll_interval))

    # ---- helpers ----
    def _file_uri_to_path(uri: str) -> Path:
        parsed = urlparse(uri)
        p = url2pathname(unquote(parsed.path or ""))
        # Windows: /C:/... -> C:/...
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return Path(p)

    def _collapse_dup_image_dir(p: Path) -> Path:
        s = str(p).replace("\\", "/")
        low = s.lower()
        if "/image/image/" in low:
            parts = s.split("/")
            lowered = [x.lower() for x in parts]
            for i in range(len(parts) - 2):
                if lowered[i] == "image" and lowered[i + 1] == "image":
                    collapsed = "/".join(parts[: i + 1] + parts[i + 2 :])
                    p1 = Path(collapsed)
                    if p1.exists():
                        return p1
                    p2 = Path(collapsed.replace("/", "\\"))
                    if p2.exists():
                        return p2
                    break
        return p

    def _read_local_file_bytes(p: str) -> tuple[bytes, str]:
        raw = (p or "").strip()
        # file:// uri
        if raw.startswith("file://"):
            path = _file_uri_to_path(raw)
        else:
            path = Path(raw)

        path = _collapse_dup_image_dir(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        return data, mime

    def _image_content_from_url(url: str) -> Dict[str, str]:
        url = (url or "").strip()
        if not url:
            raise ValueError("Empty image url")

        # data url
        if url.startswith("data:image/"):
            m = re.match(r"^data:(image/[^;]+);base64,(.+)$", url, flags=re.DOTALL)
            if not m:
                raise ValueError("Invalid data:image base64 url")
            mime, b64data = m.group(1), m.group(2)
            return {"type": "image", "data": b64data, "mime_type": mime}

        # http(s) url -> download (keep behavior consistent with gemini_chat)
        if url.startswith("http://") or url.startswith("https://"):
            with urlopen(url) as r:
                b = r.read()
                mime = r.headers.get_content_type() if hasattr(r, "headers") else None
                if not mime or not str(mime).startswith("image/"):
                    mime = "image/jpeg"
                return {"type": "image", "data": base64.b64encode(b).decode("utf-8"), "mime_type": str(mime)}

        # file:// or local path
        b, mime = _read_local_file_bytes(url)
        return {"type": "image", "data": base64.b64encode(b).decode("utf-8"), "mime_type": mime}

    def _text_content(s: str) -> Dict[str, str]:
        return {"type": "text", "text": str(s or "")}

    # ---- normalize legacy image_paths injection (optional) ----
    msgs = list(messages or [])
    image_paths = extra_config.pop("image_paths", None)
    if image_paths:
        # append images to last user message like gemini_chat
        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if (msgs[i].get("role") or "user").strip().lower() == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            msgs.append({"role": "user", "content": ""})
            last_user_idx = len(msgs) - 1

        c = msgs[last_user_idx].get("content", "")
        blocks = c if isinstance(c, list) else [{"type": "text", "text": str(c or "")}]
        for p in image_paths:
            blocks.append({"type": "image_url", "image_url": {"url": str(p)}})
        msgs[last_user_idx]["content"] = blocks

    # ---- extract system instruction ----
    system_instruction = None
    non_system_msgs: List[Dict[str, Any]] = []
    for msg in msgs:
        role_raw = (msg.get("role") or "user").strip().lower()
        if role_raw == "system":
            c = msg.get("content", "")
            if isinstance(c, str):
                system_instruction = c
            elif isinstance(c, list):
                ts = []
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        ts.append(str(b.get("text", "")))
                system_instruction = "\n".join([t for t in ts if t]).strip()
            continue
        non_system_msgs.append(msg)

    # ---- convert messages -> interactions input (array of Content objects) ----
    input_items: List[Dict[str, Any]] = []

    def _convert_content_to_items(content: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type")
                if t == "text":
                    items.append(_text_content(block.get("text", "")))
                elif t == "image_url":
                    u = (block.get("image_url") or {}).get("url")
                    if isinstance(u, str) and u.strip():
                        items.append(_image_content_from_url(u.strip()))
        else:
            items.append(_text_content(str(content or "")))
        return items

    if len(non_system_msgs) == 1 and (non_system_msgs[0].get("role") or "user").strip().lower() == "user":
        input_items.extend(_convert_content_to_items(non_system_msgs[0].get("content", "")))
    else:
        for msg in non_system_msgs:
            role_raw = (msg.get("role") or "user").strip().lower()
            role_tag = "USER" if role_raw == "user" else "ASSISTANT"
            input_items.append(_text_content(f"\n\n[{role_tag}]\n"))
            input_items.extend(_convert_content_to_items(msg.get("content", "")))

    # ---- build request ----
    # Per API: for agent runs, use agent_config (NOT generation_config).
    agent_cfg = {"type": "deep-research"}
    if thinking_summaries in ("auto", "none"):
        agent_cfg["thinking_summaries"] = thinking_summaries

    # Drop model-only config keys if caller passed them (avoid INVALID_ARGUMENT for agent runs)
    extra_config.pop("temperature", None)
    extra_config.pop("max_output_tokens", None)

    # --- Fallback for older SDKs (no client.interactions) ---
    if not hasattr(client, "interactions"):
        try:
            return _rest_create_and_poll()
        except Exception as e:
            print(f"[Gemini Deep Research Error] {e}")
            import traceback
            traceback.print_exc()
            if raise_on_error:
                raise
            return ""


    try:
        interaction = client.interactions.create(
            input=input_items if len(input_items) > 1 else (input_items[0].get("text") if input_items and input_items[0].get("type") == "text" else input_items),
            agent=agent,
            system_instruction=system_instruction,
            background=bool(background),
            store=bool(store),
            stream=bool(stream),
            agent_config=agent_cfg,
            **extra_config,
        )

        # If not background, it may already be completed.
        if not background:
            outputs = getattr(interaction, "outputs", None) or []
            for out in reversed(outputs):
                t = getattr(out, "text", None)
                if isinstance(t, str) and t.strip():
                    return t
            return getattr(interaction, "text", "") or ""

        # Background polling
        start = time.time()
        interaction_id = getattr(interaction, "id", None)
        if not interaction_id:
            # best effort fallback
            outputs = getattr(interaction, "outputs", None) or []
            return getattr(outputs[-1], "text", "") if outputs else ""

        while True:
            cur = client.interactions.get(interaction_id)
            status = (getattr(cur, "status", None) or "").lower()

            if status == "completed":
                outputs = getattr(cur, "outputs", None) or []
                for out in reversed(outputs):
                    t = getattr(out, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t
                return getattr(cur, "text", "") or ""

            if status in ("failed", "cancelled", "canceled", "error"):
                err = getattr(cur, "error", None)
                msg = ""
                if isinstance(err, dict):
                    msg = err.get("message") or str(err)
                else:
                    msg = str(err) if err is not None else ""
                raise RuntimeError(f"Deep Research interaction {interaction_id} {status}: {msg}".strip())

            if (time.time() - start) > float(max_wait_s):
                raise TimeoutError(f"Deep Research interaction timeout after {max_wait_s}s (id={interaction_id})")

            time.sleep(float(poll_interval))

    except Exception as e:
        print(f"[Gemini Deep Research Error] {e}")
        import traceback
        traceback.print_exc()
        if raise_on_error:
            raise
        return ""




def gemini_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: int = 20480,
    **extra_config: Any,
) -> str:
    client = get_gemini_client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **extra_config,
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[Gemini Text Error] {e}")
        return ""


def gemini_mm_qa(
    image_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    **extra_config: Any,
) -> str:
    try:
        from PIL import Image
    except ImportError:
        return "Error: PIL not installed"

    client = get_gemini_client(api_key)
    img = Image.open(image_path)

    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **extra_config,
    )

    contents = [img, question]

    try:
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[Gemini MM Error] {e}")
        return ""


# =========================
#   Azure OpenAI (OpenAI Python SDK)
# =========================

# NOTE:
# - Azure OpenAI 的 model 参数通常填 *deployment name*（你在 Azure Portal/Fabric/Foundry 里给 deployment 起的名字）。
# - Azure v1 API：推荐使用 OpenAI(base_url=.../openai/v1/)，无需 api-version。
# - legacy：需要 AzureOpenAI(azure_endpoint + api_version)。

# ---- Azure OpenAI (OpenAI Python SDK) ----
# Always define these names to avoid NameError even if import fails.
OpenAI = None  # type: ignore
AzureOpenAI = None  # type: ignore

try:
    # Newer OpenAI Python SDK (v1.x) exposes both clients.
    from openai import OpenAI as _OpenAI, AzureOpenAI as _AzureOpenAI  # type: ignore
    OpenAI = _OpenAI  # type: ignore
    AzureOpenAI = _AzureOpenAI  # type: ignore
except Exception:
    # Leave as None; callers will raise a helpful ImportError.
    pass


_AZURE_API_KEY: Optional[str] = None
_AZURE_BASE_URL: Optional[str] = None
_AZURE_ENDPOINT: Optional[str] = None
_AZURE_API_VERSION: Optional[str] = None


def set_azure_openai_api_key(key: str) -> None:
    global _AZURE_API_KEY
    _AZURE_API_KEY = key


def set_azure_openai_base_url(base_url: str) -> None:
    global _AZURE_BASE_URL
    _AZURE_BASE_URL = base_url


def set_azure_openai_endpoint(endpoint: str) -> None:
    global _AZURE_ENDPOINT
    _AZURE_ENDPOINT = endpoint


def set_azure_openai_api_version(version: str) -> None:
    global _AZURE_API_VERSION
    _AZURE_API_VERSION = version


def _get_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v or None


def get_azure_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Any:
    """返回 Azure OpenAI client。

    优先走 v1（OpenAI + base_url=.../openai/v1/）；否则走 legacy（AzureOpenAI + api_version）。
    """
    if OpenAI is None and AzureOpenAI is None:
        raise ImportError(
            "未找到 openai 包,请先安装:\n"
            "    pip install openai\n"
        )

    key = api_key or _AZURE_API_KEY or _get_env("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Azure OpenAI API key 未配置。请设置环境变量 AZURE_OPENAI_API_KEY。")

    bu = base_url or _AZURE_BASE_URL or _get_env("AZURE_OPENAI_BASE_URL")
    if bu:
        if not bu.endswith("/"):
            bu = bu + "/"
        return OpenAI(api_key=key, base_url=bu)

    ep = azure_endpoint or _AZURE_ENDPOINT or _get_env("AZURE_OPENAI_ENDPOINT")
    ver = api_version or _AZURE_API_VERSION or _get_env("AZURE_OPENAI_API_VERSION")
    if not ep or not ver:
        raise RuntimeError(
            "Azure OpenAI legacy 模式需要 AZURE_OPENAI_ENDPOINT 和 AZURE_OPENAI_API_VERSION。\n"
            "更推荐设置 AZURE_OPENAI_BASE_URL=.../openai/v1/ 走 v1。"
        )
    return AzureOpenAI(api_key=key, azure_endpoint=ep, api_version=ver)


def azure_chat(
    messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 4096,
    **kwargs: Any,
) -> str:
    """Azure OpenAI Chat（OpenAI Python SDK）。"""
    # Strip Gemini-only multimodal kwargs that Azure does not support
    kwargs.pop("image_paths", None)
    client = get_azure_openai_client(
        api_key=api_key,
        base_url=base_url,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    msgs = []
    for m in messages or []:
        role = (m.get("role") or "user").strip().lower()
        if role == "model":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        msgs.append({"role": role, "content": m.get("content", "")})

    def _call(params: Dict[str, Any]) -> str:
        resp = client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""

    base_params: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        **kwargs,
    }

    # 1) 先按传统 max_tokens 试
    params1 = dict(base_params)
    params1["max_tokens"] = int(max_output_tokens)

    try:
        return _call(params1)
    except Exception as e:
        msg = str(e)

        # 2) 如果服务端提示用 max_completion_tokens，则改用它重试
        if "max_completion_tokens" in msg and "max_tokens" in msg:
            params2 = dict(base_params)
            params2["max_completion_tokens"] = int(max_output_tokens)
            try:
                return _call(params2)
            except TypeError:
                # SDK 太旧不接受该参数：最后退回不传上限（保证能跑）
                try:
                    return _call(dict(base_params))
                except Exception as e2:
                    print(f"[Azure Chat Error] {e2}")
                    return ""
            except Exception as e2:
                print(f"[Azure Chat Error] {e2}")
                return ""

        # 3) 反向兼容：如果某些环境不接受 max_completion_tokens，提示用 max_tokens
        if "max_tokens" in msg and "max_completion_tokens" in msg and "Extra inputs" in msg:
            try:
                return _call(params1)
            except Exception as e2:
                print(f"[Azure Chat Error] {e2}")
                return ""

        print(f"[Azure Chat Error] {e}")
        return ""



def azure_json(
    prompt: str,
    json_schema: Dict[str, Any],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
    **extra_config: Any,
) -> str:
    """在 Azure OpenAI 上尽力产出 JSON 字符串（不强依赖结构化输出特性）。"""
    schema_text = json.dumps(json_schema, ensure_ascii=False)
    sys = system_instruction or "You are a strict JSON generator. Output only valid JSON, nothing else."
    user = (
        f"Return ONLY a JSON object that matches this JSON Schema:\n{schema_text}\n\n"
        f"Prompt:\n{prompt}\n"
    )
    return azure_chat(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        api_key=api_key,
        base_url=base_url,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **extra_config,
    )


def llm_json(
    prompt: str,
    json_schema: Dict[str, Any],
    model: str,
    provider: str = "gemini",
    **kwargs: Any,
) -> str:
    """统一 JSON 输出接口：根据 provider 调用 Gemini 或 Azure。"""
    p = (provider or "gemini").strip().lower()
    if p in {"azure", "azure_openai", "azure-openai"}:
        return azure_json(prompt=prompt, json_schema=json_schema, model=model, **kwargs)
    return gemini_json(prompt=prompt, json_schema=json_schema, model=model, **kwargs)


if __name__ == "__main__":
    pass
