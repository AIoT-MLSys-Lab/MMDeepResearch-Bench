# OCR_CHART.py
"""
轻量级 OCR-Chart-Text Match 指标:
给一个图表截图和若干段落,计算它们的匹配程度。

核心指标: ocr_chart_match_score
- 由 LLM 从段落中抽出"关于这张图/表的陈述"
- 让多模态 LLM(看图+看文字)逐条判断是否被图/表支持
- 用加权准确率汇总成一个 0~1 的分数
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

import numpy as np

# === 尝试导入 API_KEY ===
try:
    from .api import API_KEY
except ImportError:
    # 如果没有 api.py,尝试从环境变量读取,或留空让用户报错
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

from google import genai
from google.genai.types import GenerateContentConfig, Part


# ======================================================================
#                         Gemini client 封装
# ======================================================================

_CLIENT: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """懒加载一个全局 Gemini client。"""
    global _CLIENT
    if _CLIENT is None:
        if not API_KEY:
            raise RuntimeError("未找到 API_KEY! 请在 api.py 中定义 API_KEY 或设置环境变量 GEMINI_API_KEY。")
        _CLIENT = genai.Client(api_key=API_KEY)
    return _CLIENT


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }
    return mapping.get(ext, "image/png")


def _call_mm_llm(
    image_path: str,
    prompt: str,
    temperature: float = 0.1,
    max_output_tokens: int = 2048,
) -> str:
    """
    多模态 LLM 调用: 传入一张图 + 文本 prompt, 要求返回 JSON。
    """
    client = _get_client()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = _guess_mime_type(image_path)

    cfg = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",  # 直接要 JSON
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                Part.from_bytes(data=img_bytes, mime_type=mime),
                prompt,
            ],
            config=cfg,
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        print(f"[MM LLM Error] {e}")
        return "[]"


def _parse_json_list_of_dicts(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return []

    raw = raw.replace("```json", "").replace("```", "").strip()

    # 1) 直接尝试整体解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and "claims" in data and isinstance(data["claims"], list):
            return [d for d in data["claims"] if isinstance(d, dict)]
    except Exception:
        pass

    # 2) 再退而求其次: 截取第一个 '[' 到最后一个 ']' 之间的内容
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            json_str = raw[start : end + 1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass

    return []



# ======================================================================
#                     OCR-Chart – Text Match 指标
# ======================================================================

def _build_ocrchart_eval_prompt(
    paragraphs: List[str],
    max_claims: int = 15,
) -> str:
    """
    构造 prompt:
    让多模态 LLM 在"看图+看段落"的情况下,
    同时完成(抽取关于图/表的陈述 + 判定是否被图/表支持)。
    """
    text_block = "\n".join(paragraphs or [])

    return f"""
You are a careful analyst that checks whether a CHART or TABLE IMAGE matches the surrounding text.

You are given:
1) An image which is a chart or table (possibly a screenshot).
2) One or more paragraphs that supposedly describe or use this chart/table.

Your tasks:
1. From the paragraphs, extract up to {max_claims} factual claims THAT ARE SPECIFICALLY ABOUT THIS IMAGE.
   - Focus on: which series/category has which value, label, outcome or property,
     or any specific relationships/comparisons that could be verified from the image.
   - Ignore generic commentary like "the chart shows the trends" unless it states
     something concrete that can be checked in the image.

2. For each extracted claim, check whether it is supported by the IMAGE CONTENT.
   - If the image clearly supports the claim, mark it as supported.
   - If the image contradicts the claim, or the claim cannot be verified from the image,
     mark it as not supported.

3. Return ONLY a JSON array. Each element MUST be an object with:
   - "claim"       : string, a short English paraphrase of the claim.
   - "is_supported": boolean, true if the chart/table image supports the claim, false otherwise.
   - "confidence"  : float between 0 and 1 for your judgment.
   - "reason"      : short explanation mentioning relevant regions/labels/cells in the image.

IMPORTANT:
- Base your judgment ONLY on the given chart/table IMAGE, not on outside knowledge.
- If OCR is imperfect or the image is blurry, use your best guess but lower the confidence.
- If you are not sure, set "is_supported": false and explain that it cannot be verified.

[PARAGRAPHS]
{text_block}
""".strip()


def evaluate_ocr_chart(
    image_path: str,
    paragraphs: List[str],
    max_claims: int = 15,
) -> Dict[str, Any]:
    """
    核心接口: 给一张截图 chart/table 和若干段落,
    返回一个结构化结果,其中包含单个指标 ocr_chart_match_score。

    返回示例:
    {
        "score": 0.8,
        "n_total": 10,
        "n_supported": 8,
        "claims": [
            {
                "claim": "...",
                "is_supported": true,
                "confidence": 0.9,
                "reason": "..."
            },
            ...
        ]
    }
    """
    if not image_path:
        return {
            "score": None,
            "n_total": 0,
            "n_supported": 0,
            "no_image": True,
            "claims": [],
        }

    prompt = _build_ocrchart_eval_prompt(paragraphs, max_claims=max_claims)
    raw = _call_mm_llm(image_path, prompt, temperature=0.1, max_output_tokens=4096)
    claims = _parse_json_list_of_dicts(raw)

    if not claims:
        return {
            "score": 0.0,
            "n_total": 0,
            "n_supported": 0,
            "no_claims": True,
            "claims": [],
            "raw_response": raw,
        }

    norm_claims: List[Dict[str, Any]] = []
    for c in claims:
        claim_text = str(c.get("claim", "")).strip()
        is_sup = bool(c.get("is_supported", False))

        try:
            conf = float(c.get("confidence", 1.0))
        except Exception:
            conf = 1.0
        conf = max(min(conf, 1.0), 0.0)

        reason = str(c.get("reason", "")).strip()

        norm_claims.append(
            {
                "claim": claim_text,
                "is_supported": is_sup,
                "confidence": conf,
                "reason": reason,
            }
        )

    n_total = len(norm_claims)
    if n_total == 0:
        return {
            "score": 0.0,
            "n_total": 0,
            "n_supported": 0,
            "claims": [],
        }

    correct = np.array([1.0 if c["is_supported"] else 0.0 for c in norm_claims], dtype=float)
    weights = np.array([c["confidence"] for c in norm_claims], dtype=float)

    if weights.sum() <= 1e-8:
        score = float(correct.mean())
    else:
        score = float((correct * weights).sum() / weights.sum())

    return {
        "score": round(score, 4),
        "n_total": int(n_total),
        "n_supported": int(correct.sum()),
        "claims": norm_claims,
    }


# ======================================================================
#                           简单自测入口
# ======================================================================

if __name__ == "__main__":
    # 这里随便放一个截图 chart/table 的路径做自测
    test_image = "ocr.png"  # 换成你自己的截图路径

    test_paragraphs = [
    # 真实陈述（应判定为 supported）
    "表 1 展示了 Rain200L、Rain200H、DID-Data、DDN-Data 和 SPA-Data 五个数据集上各方法的 PSNR 和 SSIM 对比，其中 DRSformer 在所有数据集上的 PSNR 都是最高的。",
    "在 Rain200L 数据集上，DRSformer 的 PSNR 为 41.23 dB，高于 Restormer 的 40.99 dB 和 IDT 的 40.74 dB。",
    "在 DID-Data 上，Uformer 的 PSNR 为 35.02 dB，低于 Restormer 的 35.29 dB，也低于 DRSformer 的 35.35 dB。",
    "在 SPA-Data 上，传统的先验方法 DSC 和 GMM 的 PSNR 只有三十多 dB，明显低于 CNN 和 Transformer 系的方法。",

    # 故意错误的陈述（应判定为 not supported）
    "在 Rain200H 数据集上，DRSformer 不仅在 PSNR 上最好，而且 SSIM 也明显高于 IDT 和 Restormer。",
    "在 SPA-Data 上，DRSformer 在 PSNR 和 SSIM 两个指标上都取得了所有方法中的第一名。",
]


    print(f">>> Testing OCR-Chart Match on image: {test_image}")
    print(f">>> API KEY Present: {bool(API_KEY)}")

    try:
        res = evaluate_ocr_chart(test_image, test_paragraphs)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"\nFinal OCR-Chart Match Score: {res['score']}")
    except Exception as e:
        print(f"运行出错: {e}")