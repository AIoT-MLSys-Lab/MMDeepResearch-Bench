# TABLE.py
"""
轻量级 Table-Text Match 指标:
给一个概念/对比表(markdown 或结构化文本)和若干段落,计算它们的匹配程度。

核心指标: table_match_score
- 由 LLM 从段落中抽出"关于表的陈述"
- 对每条陈述判定是否被表支持
- 用加权准确率汇总成一个 0~1 的分数
"""

from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Any, Optional

import numpy as np

# === 尝试导入 API_KEY ===
try:
    from .api import API_KEY
except ImportError:
    # 如果没有 api.py,尝试从环境变量读取,或留空让用户报错
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

from google import genai
from google.genai.types import GenerateContentConfig


# ======================================================================
#                         Gemini client 封装
# ======================================================================

_CLIENT: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """懒加载一个全局 Gemini client。"""
    global _CLIENT
    if _CLIENT is None:
        if not API_KEY:
            raise RuntimeError("未找到 API_KEY!请在 api.py 中定义 API_KEY 或设置环境变量 GEMINI_API_KEY。")
        _CLIENT = genai.Client(api_key=API_KEY)
    return _CLIENT


def _call_text_llm(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    """纯文本 LLM 调用。"""
    client = _get_client()
    cfg = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json"  # 强制让 Gemini 输出 JSON
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=cfg,
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return "[]"


def _parse_json_list_of_dicts(raw: str) -> List[Dict[str, Any]]:
    """
    尝试把 LLM 输出解析成 list[dict]。
    支持有说明文字的情况,会截取中间的 [ ... ]。
    """
    raw = raw.strip()
    if not raw:
        return []

    # 移除可能存在的 Markdown 代码块标记
    raw = raw.replace("```json", "").replace("```", "")

    # 尝试寻找列表的起止位置
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start : end + 1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass

    # 如果截取失败,尝试直接解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass

    return []


# ======================================================================
#                           Table-Text Match
# ======================================================================

def _build_table_eval_prompt(
    table_markdown: str,
    paragraphs: List[str],
    max_claims: int = 15,
) -> str:
    """
    构造 prompt:
    让 LLM 同时"抽出关于表的陈述 + 判定是否被表支持",并输出 JSON。
    """
    text_block = "\n".join(paragraphs or [])

    return f"""
You are a careful analyst that checks whether a data table matches the surrounding text.

You are given:
1) A table (in Markdown or text format).
2) One or more paragraphs that supposedly describe or use this table.

Your tasks:
1. From the paragraphs, extract up to {max_claims} factual claims THAT ARE SPECIFICALLY ABOUT THE TABLE CONTENT.
   - Focus on: which category has which property, specific values, relationships, or comparisons found in the table.
   - Ignore generic commentary like "Table 2 shows the results" unless it describes specific data.

2. For each extracted claim, check whether it is supported by the provided table.
   - If the table clearly supports the claim, mark it as supported.
   - If the table contradicts the claim, or the claim cannot be verified from the table, mark it as not supported.

3. Return ONLY a JSON array. Each element MUST be an object with:
   - "claim"       : string, a short English paraphrase of the claim.
   - "is_supported": boolean, true if the table supports the claim, false otherwise.
   - "confidence"  : float between 0 and 1 for your judgment.
   - "reason"      : short explanation mentioning relevant rows/columns/cells.

IMPORTANT:
- Base your judgment ONLY on the table content provided below.
- If the table is messy (OCR text), try your best to interpret the structure.

[TABLE CONTENT]
```text
{table_markdown}
```

[PARAGRAPHS]
{text_block}
"""


def evaluate_concept_table(
    table_markdown: str,
    paragraphs: List[str],
    max_claims: int = 15,
) -> Dict[str, Any]:
    """
    核心接口:给一张概念/对比表(markdown)和若干段落,
    返回一个结构化结果,其中包含单个指标 table_match_score。
    """
    if not table_markdown or not table_markdown.strip():
        return {
            "score": None,
            "n_total": 0,
            "n_supported": 0,
            "no_table": True,
            "claims": [],
        }

    # 调用 LLM
    prompt = _build_table_eval_prompt(table_markdown, paragraphs, max_claims=max_claims)
    raw = _call_text_llm(prompt, temperature=0.1, max_output_tokens=4096)
    claims = _parse_json_list_of_dicts(raw)

    if not claims:
        return {
            "score": 0.0,  # 如果无法提取 claim,通常意味着相关性极低或解析失败
            "n_total": 0,
            "n_supported": 0,
            "no_claims": True,
            "claims": [],
            "raw_response": raw  # 调试用
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
        
        norm_claims.append({
            "claim": claim_text,
            "is_supported": is_sup,
            "confidence": conf,
            "reason": reason,
        })

    n_total = len(norm_claims)
    if n_total == 0:
        return {"score": 0.0, "n_total": 0, "n_supported": 0, "claims": []}

    # 计算加权分数
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
#                       Main Test Logic (Index 16)
# ======================================================================

if __name__ == "__main__":
    # 1. 填入你指定的 JSON 数据
    test_data = {
        "index": 16,
        "image_path": "",
        "paragraphs": [
            'Table 2: "月几望"在《周易》卦爻辞中的比较分析\n卦名\n爻辞\n语境含义\n道德启示\n结果\n䷈ 小畜 (上九)\n月几望,君子征\n凶。\n积累达到顶峰;\n进一步行动即为\n过度。\n警示: 在潜力最\n大时避免行动。\n凶 \n(Inauspicious)\n䷵ 归妹 (上六)\n月几望,吉。\n婚姻及时,恰在\n周期顶峰之前。\n及时: 行动与自\n然周期保持一致\n是恰当的。\n吉 (Auspicious)\n䷼ 中孚 (上六)\n月几望,马匹\n亡,无咎。\n诚信达到顶峰;\n轻微损失不可避\n免但无碍大局。\n中庸: 只要核心\n美德(诚信)得\n以保持,轻微挫\n折可接受。\n无咎 (No Blame)\n来源: 《周易》爻'
        ],
        "type": "datachart",
        "caption": 'Table 2: "月几望"在《周易》卦爻辞中的比较分析',
        "cross_page": False,
        "page": 3,
        "doc_type": "pdf",
        "missing_image": False,
        "type_rule": "ocrchart",
        "type_final": "datachart",
        "clip_used": False,
        "gemini_used": True,
        "cls_source": "gpt",
        "clip_pred": None,
        "clip_probs": None,
        "gpt_pred": "datachart",
        "gpt_raw": "ocrchart",
        "ocr_needed": False,
        "table_source": "none"
    }

    print(f">>> 正在测试数据 (Index: {test_data['index']})...")
    print(f">>> API KEY Present: {bool(API_KEY)}")

    # 2. 数据预处理
    # 这里的 paragraphs[0] 是表格的 OCR 结果
    raw_table_text = test_data["paragraphs"][0]

    # 构造一些描述性文本作为输入的"段落"
    input_paragraphs = [
        f"The figure shows {test_data['caption']}.",
        "根据表格,'小畜'卦的结果是凶 (Inauspicious),警告人们在潜力最大时避免行动。",  # 符合表格
        "然而,'归妹'卦显示结果也是凶,认为婚姻不及时。",  # 错误的(表格说是吉)
        "中孚卦提到马匹亡,但是无咎。"  # 符合表格
    ]

    print("\n[Input Table Text (Excerpt)]:")
    print(raw_table_text[:100].replace('\n', ' ') + "...")

    print("\n[Input Paragraphs for Matching]:")
    for p in input_paragraphs:
        print(f"- {p}")

    # 3. 运行评估
    result = evaluate_concept_table(
        table_markdown=raw_table_text, 
        paragraphs=input_paragraphs
    )

    # 4. 输出结果
    print("\n" + "="*50)
    print("MATCH SCORE RESULT")
    print("="*50)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"\nFinal Score: {result['score']}")
    print("-" * 50)

    if result['score'] is not None:
        if result['score'] > 0.6:
            print("结论: 段落描述与表格内容高度一致。")
        else:
            print("结论: 段落描述与表格内容存在冲突或不一致。")