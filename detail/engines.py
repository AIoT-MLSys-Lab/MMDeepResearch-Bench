# engines.py
import os
import re
from typing import Dict, Any, List, Callable, Tuple

try:
    from .api import gemini_chat
except Exception:
    gemini_chat = None  # type: ignore

# === Data engines ===
try:
    from .Chart import evaluate_concept_table
except Exception:
    evaluate_concept_table = None

try:
    from .OCRchart import evaluate_ocr_chart
except Exception:
    evaluate_ocr_chart = None

# === Optional VQA.py ===
_HAS_VQA = False
try:
    from .VQA import vqa_yesno_score as _vqa_yesno_score  # type: ignore
    _HAS_VQA = True
except Exception:
    _vqa_yesno_score = None  # type: ignore

# === Fallback MM ===
try:
    from .api import gemini_mm_qa
except Exception:
    gemini_mm_qa = None  # type: ignore


def _join_paragraphs(item: Dict[str, Any]) -> str:
    paras = item.get("paragraphs") or []
    if isinstance(paras, list):
        return "\n".join(str(p) for p in paras)
    return str(paras)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0


def _yesno_to_score(raw: str) -> float:
    t = (raw or "").strip().lower()
    if "yes" in t and "no" not in t:
        return 1.0
    if "no" in t and "yes" not in t:
        return 0.0
    return 0.5


def _vqa_yesno_fallback(image_path: str, statement: str) -> Tuple[float, str, None]:
    if gemini_mm_qa is None:
        return 0.0, "gemini_mm_qa_not_available", None
    prompt = (
        "Answer with ONLY one token: YES or NO.\n"
        f"Statement: {statement}\n"
        "Does the image support the statement?"
    )
    raw = gemini_mm_qa(
        image_path=image_path,
        question=prompt,
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=32,
    )
    return _yesno_to_score(raw), (raw or "").strip(), None


def _vqa_yesno(image_path: str, statement: str) -> Tuple[float, str, Any]:
    if _HAS_VQA and _vqa_yesno_score is not None:
        return _vqa_yesno_score(image_path=image_path, statement=statement)
    return _vqa_yesno_fallback(image_path=image_path, statement=statement)


# =========================================================
# Track A: Visual (photo/diagram)
# =========================================================
def run_vqa(item: Dict[str, Any]) -> Dict[str, Any]:
    caption = item.get("caption", "") or ""
    ctx = _join_paragraphs(item)
    img_path = item.get("image_path", "") or ""
    img_type = (item.get("type_final") or item.get("type") or "photo").lower()

    base_statement = caption or (ctx[:180] if ctx else "The image matches the description.")
    score = 0.0
    raw = ""
    err = None

    if img_path and os.path.exists(img_path):
        try:
            score, raw, _ = _vqa_yesno(image_path=img_path, statement=base_statement)
        except Exception as e:
            err = str(e)
    else:
        err = "Image file not found"

    score = _clamp01(score)

    metrics = {
        "semantic": score,
        "faithful": score,
        "formatting": 1.0,
        "vqa_score": score,
        "vqa_raw": raw,
    }
    if err:
        metrics["vqa_error"] = err

    return {
        "engine_name": "vqa_engine",
        "mode": "visual_eval",
        "metrics": metrics,
        "meta": {"type": img_type, "caption": caption[:60]},
    }


# =========================================================
# Track B: Data (datachart/ocrchart/table)
# =========================================================
def _run_data_eval(item: Dict[str, Any], eval_func: Callable, engine_name: str, format_base: float = 1.0) -> Dict[str, Any]:
    caption = item.get("caption", "")
    table_text = _join_paragraphs(item)
    paragraphs = item.get("explain_paragraphs") or item.get("paragraphs") or []
    img_path = item.get("image_path")

    try:
        if "ocr" in engine_name:
            if not img_path or not os.path.exists(img_path):
                res = {"score": 0.0, "n_supported": 0, "n_total": 0, "error": "Image not found"}
            else:
                res = eval_func(image_path=img_path, paragraphs=paragraphs)
        else:
            res = eval_func(table_markdown=table_text, paragraphs=paragraphs)
    except Exception as e:
        res = {"score": 0.0, "n_supported": 0, "n_total": 0, "error": f"Eval Error: {str(e)}", "claims": []}

    data_acc = _clamp01(res.get("score", 0.0))

    # 兼容 judgers.py：把 data_acc 同时映射到 semantic/faithful
    metrics = {
        "data_accuracy": data_acc,
        "semantic": data_acc,
        "faithful": data_acc,
        "formatting": _clamp01(format_base),
        "details": res,
    }

    return {
        "engine_name": engine_name,
        "mode": "data_eval",
        "metrics": metrics,
        "answer": f"Data Eval: {res.get('n_supported',0)}/{res.get('n_total',0)} supported.",
        "meta": {"caption": caption[:60]},
    }


def run_chart_qa(item: Dict[str, Any]) -> Dict[str, Any]:
    if evaluate_concept_table is None:
        return run_generic_mm_llm(item)
    return _run_data_eval(item, evaluate_concept_table, engine_name="chart_qa_engine")


def run_table_text_llm(item: Dict[str, Any]) -> Dict[str, Any]:
    if evaluate_concept_table is None:
        return run_generic_mm_llm(item)
    return _run_data_eval(item, evaluate_concept_table, engine_name="table_text_engine")


def run_table_ocr_llm(item: Dict[str, Any]) -> Dict[str, Any]:
    if evaluate_ocr_chart is None:
        return run_generic_mm_llm(item)
    return _run_data_eval(item, evaluate_ocr_chart, engine_name="ocr_chart_engine", format_base=0.9)


# =========================================================
# Fallback
# =========================================================
def run_generic_mm_llm(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "engine_name": "generic_engine",
        "mode": "visual_eval",
        "metrics": {"semantic": 0.0, "faithful": 0.0, "formatting": 0.0},
        "answer": "Generic fallback evaluation.",
    }


def run_imgpair_vqa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Judge a comparative / relational claim against two images."""
    claim = _join_paragraphs(item).strip() or (item.get("caption") or "Compare the two images.")

    imgs = item.get("local_images")
    if not isinstance(imgs, list) or len(imgs) < 2:
        imgs = [item.get("image_path") or "", item.get("image_path_b") or ""]

    img_a = str(imgs[0]) if len(imgs) > 0 else ""
    img_b = str(imgs[1]) if len(imgs) > 1 else ""

    if not (img_a and img_b and os.path.exists(img_a) and os.path.exists(img_b)):
        return {
            "engine_name": "imgpair_vqa_engine",
            "metrics": {"semantic": 0.0, "faithful": 0.0, "formatting": 0.0},
            "answer": "Missing or unresolved pair images for imgpair item.",
            "meta": {"verdict": "UNCLEAR", "confidence": 0.0},
        }

    clip_cos = None
    try:
        import torch
        from PIL import Image
        import open_clip  # type: ignore

        if not hasattr(run_imgpair_vqa, "_clip_bundle"):
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            model.eval()
            run_imgpair_vqa._clip_bundle = (model, preprocess)  # type: ignore

        model, preprocess = run_imgpair_vqa._clip_bundle  # type: ignore
        with torch.no_grad():
            ia = preprocess(Image.open(img_a).convert("RGB")).unsqueeze(0)
            ib = preprocess(Image.open(img_b).convert("RGB")).unsqueeze(0)
            fa = model.encode_image(ia)
            fb = model.encode_image(ib)
            fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = fb / fb.norm(dim=-1, keepdim=True)
            clip_cos = float((fa @ fb.T).squeeze().cpu().item())
    except Exception:
        clip_cos = None

    if gemini_chat is None:
        return {
            "engine_name": "imgpair_vqa_engine",
            "metrics": {"semantic": 0.0, "faithful": 0.0, "formatting": 0.0},
            "answer": "gemini_chat not available for image-image judging.",
            "meta": {"verdict": "UNCLEAR", "confidence": 0.0, "clip_cosine": clip_cos},
        }

    prompt = (
        "You are verifying a report claim about the RELATIONSHIP between two images.\n"
        "Image A is the first image; Image B is the second image.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "  verdict: one of SUPPORTS | CONTRADICTS | UNCLEAR\n"
        "  confidence: a number in [0,1]\n"
        "  rationale: a short explanation (<= 2 sentences)\n\n"
        f"Claim:\n{claim}\n"
    )

    raw = gemini_chat(
        messages=[{"role": "user", "content": prompt}],
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=256,
        image_paths=[img_a, img_b],
    ) or ""

    import json as _json
    verdict, conf = "UNCLEAR", 0.0
    rationale = raw.strip()

    try:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        obj = _json.loads(m.group(0) if m else raw)
        verdict = str(obj.get("verdict", "UNCLEAR")).upper().strip()
        conf = float(obj.get("confidence", 0.0))
        rationale = str(obj.get("rationale", "")).strip() or rationale
    except Exception:
        t = raw.lower()
        if "support" in t or "yes" in t:
            verdict = "SUPPORTS"
            conf = 0.6
        elif "contradict" in t or "no" in t:
            verdict = "CONTRADICTS"
            conf = 0.6
        else:
            verdict = "UNCLEAR"
            conf = 0.3

    conf = max(0.0, min(1.0, conf))

    if verdict == "SUPPORTS":
        faithful = conf
    elif verdict == "CONTRADICTS":
        faithful = 0.0
    else:
        faithful = 0.5 * conf

    semantic = faithful
    formatting = 1.0

    return {
        "engine_name": "imgpair_vqa_engine",
        "metrics": {"semantic": float(semantic), "faithful": float(faithful), "formatting": float(formatting)},
        "answer": rationale,
        "meta": {"verdict": verdict, "confidence": conf, "clip_cosine": clip_cos},
    }
