# VQA.py
"""
用 Gemini 2.5 计算 VQAScore: score = P("yes" | image, question)
引用 api.py 中的配置，避免重复填写 API_KEY。
"""

from __future__ import annotations

import os
import json
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

# === 从 api.py 导入 API_KEY ===
try:
    from .api import API_KEY
except ImportError:
    # 容错：如果找不到 api.py，提示用户
    raise RuntimeError("找不到 api.py，请确保它在当前目录下，并且里面定义了 API_KEY。")

from google import genai
from google.genai.types import GenerateContentConfig, Part

# 你现在用的是 flash 模型，这里可以按需改
MODEL_NAME = "gemini-2.0-flash"

# 全局 client 缓存
_CLIENT: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """
    懒加载一个全局 Gemini client，使用从 api.py 导入的 API_KEY。
    """
    global _CLIENT
    if _CLIENT is None:
        if not API_KEY:
            raise RuntimeError("api.py 中的 API_KEY 是空的！请先去 api.py 填入您的 Key。")

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


# ======================================================================
#                          核心：VQA yes/no
# ======================================================================

def vqa_yesno_score(
    image_path: str,
    statement: str,
    question_template: str = 'Does this figure show "{}"? Please answer yes or no.',
) -> Tuple[float, str, str]:
    """
    计算 VQA 分数的核心函数：score = P("yes" | image, question).

    返回:
        score      : float, 0~1 之间, 越大表示越倾向 "yes"
        question   : 实际发送给模型的问题（方便 debug）
        raw_answer : 模型原始 text 输出
    """
    client = _get_client()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = _guess_mime_type(image_path)

    question = question_template.format(statement)

    cfg = GenerateContentConfig(
        temperature=0.0,
        response_logprobs=True,  # 关键：开启 logprobs
        logprobs=5,
        max_output_tokens=5,
    )

    contents = [
        Part.from_bytes(data=img_bytes, mime_type=mime),
        question,
    ]

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=cfg,
        )
    except Exception as e:
        print(f"API Error: {e}")
        return 0.0, question, "Error"

    raw_answer = (getattr(resp, "text", "") or "").strip()

    yes_prob = 0.0
    no_prob = 0.0

    yes_tokens = {"yes", "true", "correct"}
    no_tokens = {"no", "false", "incorrect", "not"}

    try:
        # 尝试获取 logprobs
        if (
            resp.candidates
            and resp.candidates[0].logprobs_result
            and resp.candidates[0].logprobs_result.top_candidates
        ):
            first_token_candidates = (
                resp.candidates[0]
                .logprobs_result
                .top_candidates[0]
                .candidates
            )

            for cand in first_token_candidates:
                tok_text = cand.token.lower().strip()
                prob = float(np.exp(cand.log_probability))

                if tok_text in yes_tokens:
                    yes_prob += prob
                elif tok_text in no_tokens:
                    no_prob += prob
        else:
            # 如果拿不到 logprobs (极少数情况) → fallback 在下面 except 里处理
            pass

    except Exception as e:
        print(f"Logprobs calculation warning: {e}")
        # Fallback 到粗暴文本匹配
        lower_ans = raw_answer.lower()
        if "yes" in lower_ans:
            return 1.0, question, raw_answer
        if "no" in lower_ans:
            return 0.0, question, raw_answer
        return 0.0, question, raw_answer

    # 归一化
    total_prob = yes_prob + no_prob
    if total_prob < 1e-9:
        score = 0.0
    else:
        score = yes_prob / total_prob

    return score, question, raw_answer


# ======================================================================
#              通用 LLM 调用封装（给下面两个指标用）
# ======================================================================

def _call_text_llm(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> str:
    """
    只用文本的 LLM 调用封装（基于同一个 Gemini client）。
    """
    client = _get_client()
    cfg = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    )
    return (getattr(resp, "text", "") or "").strip()


def _call_mm_llm(
    image_path: str,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
) -> str:
    """
    图 + 文本 的通用 LLM 调用封装，用于从图里抽数字 / 标签等。
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
    )

    contents = [
        Part.from_bytes(data=img_bytes, mime_type=mime),
        prompt,
    ]

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=cfg,
    )
    return (getattr(resp, "text", "") or "").strip()


# ======================================================================
#        第一优先：Diagram 用的 Hallucination Score (CHAIR 风格)
# ======================================================================

def _parse_json_list_of_strings(raw: str) -> List[str]:
    """
    把 LLM 输出解析成 string list，先尝试 JSON，失败就按行粗暴切。
    """
    raw = raw.strip()
    if not raw:
        return []

    # 尝试直接 JSON 解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
            if out:
                return out
    except Exception:
        pass

    # 尝试截取中间的 [ ... ]
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, list):
                out = [str(x).strip() for x in data if str(x).strip()]
                if out:
                    return out
    except Exception:
        pass

    # 最兜底：逐行 split
    lines = [ln.strip("-* \t") for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def extract_diagram_entities(
    caption: str,
    paragraphs: List[str],
    max_entities: int = 10,
) -> List[str]:
    """
    从 caption + 段落中抽出「这张图里应该出现的关键对象 / 元素」名字列表。
    只做高层语义，不区分类型。
    """
    context = caption.strip() + "\n\n" + "\n".join(paragraphs or [])

    prompt = f"""
你是一个信息抽取工具。下面是一张图的图注和相关文字描述，请你提取
“这张图中应该清晰可见的关键对象或元素名称”。

要求：
1. 只抽图中应该能看到的对象/元素（例如：hotel lobby, concierge desk, guest, chandelier）。
2. 不要抽抽象概念（比如 “power”, “love” 这类看不见的）。
3. 用英文短语或简单名词短语。
4. 最多返回 {max_entities} 个。
5. 只输出 JSON 数组，形如：["hotel lobby", "concierge desk", "guest"]，不要任何多余文字。

[图注和文本]
{context}
"""

    raw = _call_text_llm(prompt, temperature=0.1, max_output_tokens=512)
    entities = _parse_json_list_of_strings(raw)

    # 去重 + 清洗
    seen = set()
    cleaned: List[str] = []
    for e in entities:
        e_clean = e.strip().strip('"').strip("'")
        if not e_clean:
            continue
        if e_clean.lower() in seen:
            continue
        seen.add(e_clean.lower())
        cleaned.append(e_clean)

    return cleaned[:max_entities]


def compute_diagram_hallucination_score(
    image_path: str,
    caption: str,
    paragraphs: List[str],
    vqa_threshold: float = 0.5,
    max_entities: int = 10,
) -> Dict[str, Any]:
    """
    Diagram 层用的 Hallucination Score：

    1. 用 LLM 从 caption+段落中抽出应该在图中出现的实体列表 entities。
    2. 对每个实体 e，问 VQA: "Does this figure show \"{e}\"?"，得到 P(yes)。
    3. 若 P(yes) < vqa_threshold，则视为 hallucinated（文字提到但图中看不到）。
    4. hallucination_rate = (# hallucinated) / (# entities)
       hallucination_score = 1 - hallucination_rate

    返回字典结构，方便第四层直接用：
    {
        "score": float,
        "hallucination_rate": float,
        "n_total": int,
        "n_hallucinated": int,
        "entities": [...],
        "per_entity": [
            {
                "name": str,
                "yes_prob": float,
                "question": str,
                "answer": str,
                "is_hallucinated": bool,
            },
            ...
        ],
    }
    """
    entities = extract_diagram_entities(caption, paragraphs, max_entities=max_entities)

    if not entities:
        # 没抽到实体 → 默认不惩罚 hallucination，但标记 no_entities
        return {
            "score": 1.0,
            "hallucination_rate": 0.0,
            "n_total": 0,
            "n_hallucinated": 0,
            "entities": [],
            "per_entity": [],
            "no_entities": True,
        }

    per_entity: List[Dict[str, Any]] = []
    n_hallu = 0

    for e in entities:
        try:
            yes_prob, q, ans = vqa_yesno_score(image_path, e)
        except Exception as ex:
            # 出错就当作不确定，不计入 hallucination，但记录错误
            per_entity.append(
                {
                    "name": e,
                    "yes_prob": None,
                    "question": None,
                    "answer": None,
                    "is_hallucinated": None,
                    "error": str(ex),
                }
            )
            continue

        is_hallu = (yes_prob is not None) and (yes_prob < vqa_threshold)
        if is_hallu:
            n_hallu += 1

        per_entity.append(
            {
                "name": e,
                "yes_prob": float(yes_prob),
                "question": q,
                "answer": ans,
                "is_hallucinated": bool(is_hallu),
            }
        )

    n_total = len([pe for pe in per_entity if pe.get("yes_prob") is not None])

    if n_total == 0:
        # 全部都出错了 / 没有有效 VQA 结果
        return {
            "score": 1.0,
            "hallucination_rate": 0.0,
            "n_total": 0,
            "n_hallucinated": 0,
            "entities": entities,
            "per_entity": per_entity,
            "no_valid_vqa": True,
        }

    hallucination_rate = n_hallu / n_total
    hallucination_score = 1.0 - hallucination_rate

    return {
        "score": float(hallucination_score),
        "hallucination_rate": float(hallucination_rate),
        "n_total": int(n_total),
        "n_hallucinated": int(n_hallu),
        "entities": entities,
        "per_entity": per_entity,
    }


# ======================================================================
#  第二优先：Diagram 轻量版 OCR–Text Consistency（数值对齐版本）
# ======================================================================

def _parse_json_number_list(raw: str) -> List[Dict[str, Any]]:
    """
    解析形如:
        [
          {"value": "3", "unit": "years"},
          {"value": "2020", "unit": "year"}
        ]
    的 JSON 数组。解析失败时返回空列表。
    """
    raw = raw.strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out: List[Dict[str, Any]] = []
            for x in data:
                if isinstance(x, dict):
                    v = str(x.get("value", "")).strip()
                    u = str(x.get("unit", "")).strip()
                    if v:
                        out.append({"value": v, "unit": u})
            return out
    except Exception:
        pass

    # 兜底：不强行从自由文本解析，直接返回空
    return []


def extract_numbers_from_text(
    paragraphs: List[str],
) -> List[Dict[str, Any]]:
    """
    从文本段落中抽取数字（年份、百分比、具体数值等）。

    返回:
        [{"value": "2020", "unit": "year"}, {"value": "3", "unit": "times"}, ...]
    """
    context = "\n".join(paragraphs or [])

    prompt = f"""
你是一个信息抽取工具。请从下面的文本中抽取所有出现的“具体数字信息”，
包括年份、百分比、次数、比例、绝对值等。

请用 JSON 数组输出，每个元素为一个对象:
  {{
    "value": "原文中的数字字符串",
    "unit": "它的单位或含义（如果没有就留空字符串）"
  }}

不要输出任何解释性文字，只输出 JSON。

[文本]
{context}
"""
    raw = _call_text_llm(prompt, temperature=0.1, max_output_tokens=512)
    return _parse_json_number_list(raw)


def extract_numbers_from_image(
    image_path: str,
) -> List[Dict[str, Any]]:
    """
    用多模态 LLM 从图中抽取可见的数字信息。
    同样返回 list[{"value": ..., "unit": ...}, ...]
    """
    prompt = """
从这张图中抽取你能看到的“重要数字信息”，包括年份、百分比、标注的数值等。

请用 JSON 数组输出，每个元素为一个对象:
  {
    "value": "图上的数字字符串",
    "unit": "对应的单位或含义（比如 %, year, people 等，如果没有就留空字符串）"
  }

不要输出任何解释，只输出 JSON。
"""
    raw = _call_mm_llm(image_path, prompt, temperature=0.0, max_output_tokens=512)
    return _parse_json_number_list(raw)


def compute_ocr_text_consistency_for_diagram(
    image_path: str,
    paragraphs: List[str],
    value_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Diagram 的轻量版 OCR–Text Consistency：
      - 从文本中抽取所有数字 text_numbers
      - 从图像中抽取数字 image_numbers
      - 看文本里每个数字在图中是否能找到“相同/近似”的值，命中率越高，score 越高。

    返回:
    {
        "score": float or None,  # None 表示没有足够的数字做判断
        "n_text_numbers": int,
        "n_image_numbers": int,
        "n_matched": int,
        "text_numbers": [...],
        "image_numbers": [...],
    }
    """
    text_nums = extract_numbers_from_text(paragraphs)
    img_nums = extract_numbers_from_image(image_path)

    n_text = len(text_nums)
    n_img = len(img_nums)

    if n_text == 0 or n_img == 0:
        # 没有数字/无法解析 → 没法计算，返回 None
        return {
            "score": None,
            "n_text_numbers": n_text,
            "n_image_numbers": n_img,
            "n_matched": 0,
            "text_numbers": text_nums,
            "image_numbers": img_nums,
            "not_enough_numbers": True,
        }

    def _to_float_safe(s: str) -> Optional[float]:
        s = s.replace(",", "").strip()
        try:
            # 去掉末尾的 % 这种
            if s.endswith("%"):
                return float(s[:-1]) / 100.0
            return float(s)
        except Exception:
            return None

    matched = 0

    for t in text_nums:
        v_text = _to_float_safe(t.get("value", ""))
        if v_text is None:
            continue

        # 在 image_numbers 里找一个“近似”的值
        found = False
        for im in img_nums:
            v_img = _to_float_safe(im.get("value", ""))
            if v_img is None:
                continue

            # 相对误差判断
            denom = max(abs(v_text), 1e-6)
            rel_err = abs(v_text - v_img) / denom
            if rel_err <= value_tolerance:
                found = True
                break

        if found:
            matched += 1

    score = matched / max(1, n_text)

    return {
        "score": float(score),
        "n_text_numbers": n_text,
        "n_image_numbers": n_img,
        "n_matched": int(matched),
        "text_numbers": text_nums,
        "image_numbers": img_nums,
    }


# ======================================================================
#                           简单自测入口
# ======================================================================

if __name__ == "__main__":
    # 这里的路径记得改成你本地实际存在的路径
    test_image = "split_output_final\\apple.png"
    test_statement = "The Red Fuji Apple"
    test_paragraphs = [
        '''The Red Fuji is widely considered one of the world’s most premier apple varieties, distinguished by its impressive size and beautiful, bi-colored skin that features streaks of deep crimson over a yellow-green background.

What truly sets the Red Fuji apart is its texture and flavor profile. Beneath its skin lies a dense, creamy-white flesh that is exceptionally crisp and juicy. Unlike softer varieties, the Fuji offers a satisfying "snap" when bitten into. Flavor-wise, it is celebrated for its intense sweetness—often boasting a higher sugar content (Brix level) than almost any other apple—balanced by a very low acidity.

Originating in Japan as a cross between the Ralls Janet and the Red Delicious, the Red Fuji also has an incredible shelf life. It can remain fresh and crunchy for months with proper refrigeration, making it a favorite for snacking, salads, and gifting.''',
    ]

    print(">>> Test VQAScore (using API_KEY from api.py)")
    try:
        score, q, ans = vqa_yesno_score(test_image, test_statement)
        print("Question :", q)
        print("Answer   :", ans)
        print("VQAScore :", score)
    except Exception as e:
        print(f"VQA 运行出错: {e}")

    print("\n>>> Test Diagram Hallucination Score")
    try:
        hallu = compute_diagram_hallucination_score(
            test_image,
            caption=test_statement,
            paragraphs=test_paragraphs,
        )
        print(json.dumps(hallu, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Hallucination 运行出错: {e}")

    print("\n>>> Test OCR-Text Consistency (Diagram light version)")
    try:
        ocrc = compute_ocr_text_consistency_for_diagram(
            test_image,
            paragraphs=test_paragraphs,
        )
        print(json.dumps(ocrc, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"OCR-Text Consistency 运行出错: {e}")
