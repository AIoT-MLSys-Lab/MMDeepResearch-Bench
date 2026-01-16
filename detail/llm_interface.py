"""
Abstraction layer for plugging different deep research LLMs.

Fixed version with consistent image handling logic.
Hardened version:
- deep-copies messages to avoid cross-thread mutation
- never silently drops images when image_paths were provided (raises instead)
- optional downscale+JPEG recompress when images exceed byte limit
- safer vision drop patterns (remove overly broad "mini")
- basic debug stats for image injection
"""

from __future__ import annotations

from abc import ABC
from typing import List, Dict, Any, Optional, Tuple
import os
import time
import random
import base64
import mimetypes
import copy
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname
from io import BytesIO

from openai import OpenAI, AzureOpenAI

from .api import gemini_chat, gemini_deep_research

try:
    from PIL import Image  # optional, but recommended
except Exception:
    Image = None

Message = Dict[str, Any]


# =========================
# Base Interface
# =========================
class BaseLLMClient(ABC):
    supports_vision: bool = False

    def __init__(self):
        self._vision_enabled = self.supports_vision and not self._should_drop_images()

    def _should_drop_images(self) -> bool:
        return False

    @property
    def vision_enabled(self) -> bool:
        return self._vision_enabled

    def _strip_images_from_messages(self, messages: List[Message]) -> List[Message]:
        new_messages: List[Message] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                texts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        texts.append(str(b.get("text", "")))
                new_messages.append({"role": m.get("role"), "content": "\n".join(t for t in texts if t).strip()})
            else:
                new_messages.append(m)
        return new_messages

    def _preprocess_messages_and_images(
        self,
        messages: List[Message],
        image_paths: Optional[List[str]],
    ) -> Tuple[List[Message], List[str]]:
        if image_paths is None:
            image_paths = []
        elif isinstance(image_paths, str):
            image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            image_paths = []

        valid_paths = [str(p).strip() for p in image_paths if p and str(p).strip()]

        # 只有 vision 不可用时才剥图
        if not self.vision_enabled:
            return self._strip_images_from_messages(messages), []

        # vision 可用但没传 image_paths：保留 messages（可能本来就含图）
        if not valid_paths:
            return messages, []

        return messages, valid_paths


# =========================
# Dummy
# =========================
class DummyEchoClient(BaseLLMClient):
    supports_vision = False

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        kwargs.pop("image_paths", None)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        return f"[DUMMY REPLY] Received: {str(last_user)[:50]}..."


# =========================
# Shared Vision Helper
# =========================
class _VisionMixin:
    """
    Helper to inject local/remote images into the LAST user message,
    as OpenAI-compatible content blocks.

    Hardened:
    - optional downscale/recompress when too large
    - option to raise if image_paths were provided but none injected
    """

    def _file_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        p = url2pathname(unquote(parsed.path or ""))
        # Windows: /C:/... -> C:/...
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return Path(p)

    def _normalize_image_input_to_path(self, p: str) -> Path:
        """
        Normalize to local Path. Fix common ".../image/image/..." duplication.
        """
        p = (p or "").strip()
        if not p:
            return Path("")

        if p.startswith("file://"):
            path = self._file_uri_to_path(p)
        else:
            path = Path(p)

        try:
            if not path.is_absolute():
                path = path.resolve()
        except Exception:
            pass

        s = str(path).replace("\\", "/")
        if "/image/image/" in s.lower():
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

        return path

    def _encode_image_bytes_to_data_url(self, img_bytes: bytes, mime: str) -> str:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _downscale_and_reencode_if_needed(
        self,
        path: Path,
        max_image_bytes: int,
        max_side: int = 1280,
        quality: int = 75,
    ) -> Tuple[bytes, str]:
        """
        Returns (bytes, mime). If PIL unavailable, raises on oversize.
        """
        raw = path.read_bytes()
        if len(raw) <= max_image_bytes:
            mime, _ = mimetypes.guess_type(str(path))
            if not mime or not mime.startswith("image/"):
                mime = "image/jpeg"
            return raw, mime

        if Image is None:
            raise ValueError(f"Image too large ({len(raw)} bytes) and PIL not installed: {path}")

        # Try downscale + JPEG recompress
        with Image.open(BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, float(max_side) / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)))

            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            out = buf.getvalue()

        # If still too large, be more aggressive
        if len(out) > max_image_bytes:
            for q in (65, 55, 45):
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                out = buf.getvalue()
                if len(out) <= max_image_bytes:
                    break

        if len(out) > max_image_bytes:
            raise ValueError(f"Image still too large after recompress ({len(out)} bytes): {path}")

        return out, "image/jpeg"

    def _encode_local_image_to_data_url(self, p: str, max_image_bytes: int) -> str:
        p = (p or "").strip()
        if p.startswith("http://") or p.startswith("https://") or p.startswith("data:"):
            return p

        path = self._normalize_image_input_to_path(p)
        if not path or not path.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        img_bytes, mime = self._downscale_and_reencode_if_needed(
            path=path,
            max_image_bytes=max_image_bytes,
        )
        return self._encode_image_bytes_to_data_url(img_bytes, mime)

    @staticmethod
    def _count_image_blocks(messages: List[Message]) -> int:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return 0
        c = last_user.get("content")
        if not isinstance(c, list):
            return 0
        return sum(1 for b in c if isinstance(b, dict) and b.get("type") == "image_url")

    def _inject_images(
        self,
        messages: List[Message],
        image_paths: List[str],
        max_image_bytes: int,
        raise_on_empty_injection: bool = False,
        debug_prefix: str = "",
    ) -> List[Message]:
        if not image_paths:
            return messages

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            if raise_on_empty_injection:
                raise RuntimeError("No user message to inject images into.")
            return messages

        new_messages: List[Message] = []
        injected = 0

        for i, m in enumerate(messages):
            if i != last_user_idx:
                new_messages.append(m)
                continue

            orig = m.get("content", "")

            # 1) keep existing blocks (text + image_url)
            blocks: List[Dict[str, Any]] = []
            if isinstance(orig, list):
                for b in orig:
                    if isinstance(b, dict) and b.get("type") in ("text", "image_url"):
                        blocks.append(b)
            else:
                blocks = [{"type": "text", "text": str(orig or "")}]

            # 2) ensure at least one text
            if not any(isinstance(b, dict) and b.get("type") == "text" for b in blocks):
                blocks.insert(0, {"type": "text", "text": ""})

            # 3) dedupe + append images
            existing_urls = set()
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "image_url":
                    u = (b.get("image_url") or {}).get("url")
                    if isinstance(u, str):
                        existing_urls.add(u)

            for p in image_paths:
                p = (p or "").strip()
                if not p:
                    continue
                url = self._encode_local_image_to_data_url(p, max_image_bytes=max_image_bytes)
                if url in existing_urls:
                    continue
                blocks.append({"type": "image_url", "image_url": {"url": url}})
                existing_urls.add(url)
                injected += 1

            new_messages.append({"role": "user", "content": blocks})

        if debug_prefix:
            print(f"{debug_prefix} injected_images={injected} total_image_blocks={self._count_image_blocks(new_messages)}")

        if raise_on_empty_injection and injected == 0:
            raise RuntimeError("image_paths provided but no images were injected (would become text-only).")

        return new_messages


# =========================
# Gemini
# =========================
class GeminiChatClient(BaseLLMClient, _VisionMixin):
    supports_vision = True

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.max_image_bytes = int(os.getenv("GEMINI_MAX_IMAGE_BYTES", 18 * 1024 * 1024))
        super().__init__()

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        messages = copy.deepcopy(messages)

        image_paths = kwargs.pop("image_paths", None)
        messages, valid_image_paths = self._preprocess_messages_and_images(messages, image_paths)

        if valid_image_paths:
            messages = self._inject_images(
                messages,
                valid_image_paths,
                max_image_bytes=self.max_image_bytes,
                raise_on_empty_injection=True,
                debug_prefix="[Gemini Vision]",
            )

        if "deep-research" in (self.model_name or "").lower():
            return gemini_deep_research(
                messages=messages,
                agent=self.model_name,
                api_key=self.api_key,
                **kwargs,
            )

        return gemini_chat(
            messages=messages,
            model=self.model_name,
            api_key=self.api_key,
            **kwargs,
        )


# =========================
# Azure OpenAI
# =========================
class AzureOpenAIChatClient(BaseLLMClient, _VisionMixin):
    supports_vision = True

    _NO_TEMPERATURE_PATTERNS = (
        "o1",
        "o3-mini",
        "o3mini",
        "gpt-5",
    )

    _NO_VISION_PATTERNS = (
        "o3-mini",
        "o3mini",
        "minio3",
        "kimi",
    )

    def __init__(
        self,
        deployment_name: str,
        api_key: Optional[str],
        azure_endpoint: Optional[str],
        api_version: Optional[str] = None,
    ):
        self.deployment_name = (deployment_name or "").strip()
        self.api_key = (api_key or "").strip().strip('"')
        self.azure_endpoint = (azure_endpoint or "").strip().strip('"')
        self.api_version = (api_version or "").strip().strip('"') or "2024-10-21"

        self.max_retries = self._safe_int(os.getenv("AZURE_MAX_RETRIES"), default=5)
        self.base_delay = self._safe_float(os.getenv("AZURE_RETRY_BASE_DELAY"), default=4.0)
        self.max_image_bytes = self._safe_int(os.getenv("AZURE_MAX_IMAGE_BYTES"), default=(8 * 1024 * 1024))

        if not self.deployment_name:
            raise ValueError("Azure client requires deployment_name.")
        if not self.api_key:
            raise ValueError("Missing Azure API Key.")
        if not self.azure_endpoint:
            raise ValueError("Missing Azure endpoint.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )

        super().__init__()

    def _should_drop_images(self) -> bool:
        name = self.deployment_name.lower().replace("_", "-").strip()
        return any(p in name for p in self._NO_VISION_PATTERNS)

    @staticmethod
    def _safe_int(v: Optional[str], default: int) -> int:
        try:
            if v is None or not str(v).strip():
                return default
            return int(str(v).strip())
        except Exception:
            return default

    @staticmethod
    def _safe_float(v: Optional[str], default: float) -> float:
        try:
            if v is None or not str(v).strip():
                return default
            return float(str(v).strip())
        except Exception:
            return default

    def _drop_temperature(self) -> bool:
        name = self.deployment_name.lower().replace("_", "-").strip()
        return any(p in name for p in self._NO_TEMPERATURE_PATTERNS)

    def _call_with_compat(self, payload: Dict[str, Any]) -> str:
        try:
            resp = self.client.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "")
        except Exception as e1:
            msg1 = str(e1)

        if "max_completion_tokens" in payload:
            p2 = dict(payload)
            mct = p2.pop("max_completion_tokens", None)
            if mct is not None:
                p2["max_tokens"] = mct
            try:
                resp = self.client.chat.completions.create(**p2)
                return (resp.choices[0].message.content or "")
            except Exception as e2:
                msg2 = str(e2)
        else:
            msg2 = msg1

        if "temperature" in payload and ("temperature" in msg1.lower() or "temperature" in msg2.lower()):
            p3 = dict(payload)
            p3.pop("temperature", None)
            try:
                resp = self.client.chat.completions.create(**p3)
                return (resp.choices[0].message.content or "")
            except Exception:
                pass

            if "max_completion_tokens" in p3:
                p4 = dict(p3)
                mct = p4.pop("max_completion_tokens", None)
                if mct is not None:
                    p4["max_tokens"] = mct
                resp = self.client.chat.completions.create(**p4)
                return (resp.choices[0].message.content or "")

        raise RuntimeError(f"Azure chat call failed: {msg1}")

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        messages = copy.deepcopy(messages)

        image_paths = kwargs.pop("image_paths", None)
        messages, valid_image_paths = self._preprocess_messages_and_images(messages, image_paths)

        if valid_image_paths:
            # IMPORTANT: do not silently strip; raise if injection fails
            messages = self._inject_images(
                messages,
                valid_image_paths,
                max_image_bytes=self.max_image_bytes,
                raise_on_empty_injection=True,
                debug_prefix="[Azure Vision]",
            )

        max_out = None
        if "max_output_tokens" in kwargs:
            try:
                max_out = int(kwargs.pop("max_output_tokens"))
            except Exception:
                pass

        temp = kwargs.pop("temperature", 0.7)
        if self._drop_temperature():
            temp = None

        kwargs.pop("candidate_count", None)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                payload: Dict[str, Any] = {"model": self.deployment_name, "messages": messages, **kwargs}
                if temp is not None:
                    payload["temperature"] = temp
                if max_out is not None:
                    payload["max_completion_tokens"] = max_out

                return self._call_with_compat(payload)

            except Exception as e:
                last_err = e
                sleep_s = min(self.base_delay * (2 ** attempt), 30.0) + random.random()
                time.sleep(sleep_s)

        raise RuntimeError(f"Azure chat failed after retries: {last_err}")


# =========================
# OpenRouter
# =========================
class OpenRouterChatClient(BaseLLMClient, _VisionMixin):
    supports_vision = True

    # Remove overly broad "mini" – it will wrongly disable vision on many VL models.
    _NO_VISION_PATTERNS = (
        "kimi",
        "o3-mini",
        "o3mini",
    )

    def __init__(self, model_name: str, api_key: Optional[str]):
        self.model_name = (model_name or "").strip()
        self.api_key = (api_key or "").strip().strip('"')

        if not self.model_name:
            raise ValueError("OpenRouter client requires model_name.")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/OpenRouter",
                "X-Title": "MMDR-Benchmark",
            },
        )

        # Use smaller default to avoid base64 request-size blowups in batch.
        self.max_image_bytes = self._safe_int(os.getenv("OPENROUTER_MAX_IMAGE_BYTES"), default=(6 * 1024 * 1024))

        super().__init__()

    def _should_drop_images(self) -> bool:
        name = self.model_name.lower().replace("_", "-").strip()
        return any(p in name for p in self._NO_VISION_PATTERNS)

    @staticmethod
    def _safe_int(v: Optional[str], default: int) -> int:
        try:
            if v is None or not str(v).strip():
                return default
            return int(str(v).strip())
        except Exception:
            return default

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        messages = copy.deepcopy(messages)

        image_paths = kwargs.pop("image_paths", None)
        messages, valid_image_paths = self._preprocess_messages_and_images(messages, image_paths)

        if valid_image_paths:
            # IMPORTANT: raise if injection ends up empty; do not silently text-fallback.
            messages = self._inject_images(
                messages,
                valid_image_paths,
                max_image_bytes=self.max_image_bytes,
                raise_on_empty_injection=True,
                debug_prefix="[OpenRouter Vision]",
            )

        if (
            self.model_name.lower() == "perplexity/sonar-deep-research"
            and "max_output_tokens" not in kwargs
            and "max_tokens" not in kwargs
        ):
            kwargs["max_tokens"] = 800000

        if "max_output_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

        kwargs.pop("candidate_count", None)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"include_reasoning": True},
            **kwargs,
        )

        return (resp.choices[0].message.content or "")
