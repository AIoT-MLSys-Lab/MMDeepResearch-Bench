# clip_classifier.py
import os
from typing import Dict, Tuple, Optional, Callable, Any, List

import torch
from PIL import Image

# 用 open_clip 而不是 clip
try:
    import open_clip
except ImportError:
    open_clip = None

# 四类标签
CLASSES = ["diagram", "datachart", "ocrchart", "photo"]

# 全局缓存
_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClipNotAvailable(Exception):
    """在本机上 CLIP 无法使用时抛出，用于上层优雅降级。"""
    pass


def get_image_embedding(image_path: str, device: Optional[str] = None):
    """Return a normalized CLIP image embedding (1D torch tensor on CPU).

    This is used for image-image consistency.
    Raises ClipNotAvailable if open_clip is unavailable or image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise ClipNotAvailable(f"image file not found: {image_path}")
    _lazy_init(device)

    img = Image.open(image_path).convert("RGB")
    image = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        feat = _MODEL.encode_image(image)  # [1, D]
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).detach().cpu()


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two normalized vectors (or unnormalized torch tensors)."""
    if a is None or b is None:
        return 0.0
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # Normalize defensively.
        a2 = a / (a.norm() + 1e-12)
        b2 = b / (b.norm() + 1e-12)
        return float(torch.dot(a2, b2).item())
    # Fallback for unexpected types.
    return 0.0


def cosine_similarity01(a, b) -> float:
    """Cosine similarity mapped to [0, 1]."""
    cos = cosine_similarity(a, b)
    # cos in [-1, 1]
    v = 0.5 * (cos + 1.0)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def mean_pairwise_cos01(embs: List[torch.Tensor]) -> float | None:
    """Mean pairwise cosine similarity (mapped to [0, 1]).

    Returns None when there are fewer than 2 embeddings.
    """
    embs = [e for e in embs if isinstance(e, torch.Tensor)]
    if len(embs) < 2:
        return None
    s = 0.0
    n = 0
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            s += cosine_similarity01(embs[i], embs[j])
            n += 1
    return None if n == 0 else float(s / n)


def coherence_to_centroid01(embs: List[torch.Tensor]) -> float | None:
    """Coherence score in [0,1] computed as mean similarity to centroid.

    Returns None when there are fewer than 2 embeddings.
    """
    embs = [e for e in embs if isinstance(e, torch.Tensor)]
    if len(embs) < 2:
        return None
    # centroid
    c = torch.stack(embs, dim=0).mean(dim=0)
    c = c / (c.norm() + 1e-12)
    sims = [cosine_similarity01(e, c) for e in embs]
    return float(sum(sims) / max(1, len(sims)))


def _lazy_init(device: Optional[str] = None):
    """
    延迟初始化 open-clip 模型，只加载一次。
    """
    global _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE

    if _MODEL is not None:
        return

    if open_clip is None:
        raise ClipNotAvailable("open_clip_torch 未安装")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device

    # 这里用你环境里比较通用的 ViT-B-32，预训练权重用 "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer


def _class_text_prompts() -> list[str]:
    """
    给四个 router 类别写一组英文 prompt，
    open-clip 会对 "image vs text prompts" 做相似度匹配。
    （后面你想加强中文语义，可以在这里把中文关键词也加进去）
    """
    return [
        # diagram
        "a diagram or schematic, technical drawing, flowchart, or architecture figure",
        # datachart
        "a data chart or plot with axes and numeric values, such as bar chart, line chart, or pie chart",
        # ocrchart
        "a table or spreadsheet or screenshot of a grid with many cells, numbers and text aligned in rows and columns",
        # photo
        "a natural photo or illustration or picture, not mainly a table or chart",
    ]


def classify_image_with_clip(image_path: str, device: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    纯 CLIP 分类接口：
      - pred_label: "diagram" / "datachart" / "ocrchart" / "photo"
      - prob_dict: {label: prob}
    如果 CLIP 不可用，会抛 ClipNotAvailable。
    """
    if not os.path.exists(image_path):
        raise ClipNotAvailable(f"image file not found: {image_path}")

    _lazy_init(device)

    # 读图
    img = Image.open(image_path).convert("RGB")
    image = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)  # [1, C, H, W]

    # 文本 prompts
    texts = _class_text_prompts()
    text_tokens = _TOKENIZER(texts).to(_DEVICE)

    with torch.no_grad():
        image_features = _MODEL.encode_image(image)        # [1, D]
        text_features = _MODEL.encode_text(text_tokens)    # [4, D]

        # L2 norm
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 余弦相似度
        logits = (image_features @ text_features.T).squeeze(0)  # [4]
        probs = logits.softmax(dim=0).cpu().numpy()

    import numpy as np
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return pred_label, prob_dict


# =====================
#   GPT / 小模型 预留
# =====================

# 小模型分类函数类型：输入 prompt，输出四个标签之一
LLMClassifierFn = Callable[[str], str]


def build_gpt_prompt(
    caption: str,
    paragraphs: Optional[List[str]] = None,
) -> str:
    """
    构造发给 GPT 小模型的 prompt（纯文本）。
    这里假设模型只是看文字，不看图片本身；
    如果你用多模态 GPT，可以在外层把图片同时传进去。
    """
    para_text = "\n".join(paragraphs or [])
    prompt = f"""你是一个文档图像分类器，需要把一张图分到四个类别之一：

1) diagram：示意图 / 结构图 / 概念图 / 流程图 / 拓扑图
2) datachart：带坐标轴和数据的图表，例如折线图、柱状图、饼图等
3) ocrchart：看起来是表格截图或类似 Excel 的网格（很多行列）
4) photo：自然照片、插画、摄影画面等

下面是这张图在文档中的说明文字（caption）和附近的段落内容，
请只输出一个类别标识：diagram / datachart / ocrchart / photo。

[Caption]
{caption}

[Context paragraphs]
{para_text}
"""
    return prompt


def normalize_llm_label(raw: str) -> Optional[str]:
    """
    把小模型返回的字符串兜底映射到四个合法标签之一。
    """
    if not raw:
        return None
    s = raw.strip().lower()

    # 常见写法兜底
    if "diagram" in s or "示意" in s or "结构图" in s:
        return "diagram"
    if "data" in s or "chart" in s or "图表" in s or "plot" in s:
        return "datachart"
    if "table" in s or "表格" in s or "ocr" in s or "spreadsheet" in s:
        return "ocrchart"
    if "photo" in s or "图片" in s or "照片" in s or "image" in s:
        return "photo"

    # 如果直接就是四个标签之一
    if s in CLASSES:
        return s

    return None


def classify_image_smart(
    image_path: str,
    caption: str = "",
    paragraphs: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_clip: bool = True,
    gpt_fn: Optional[LLMClassifierFn] = None,
    prefer: str = "clip",
) -> Tuple[str, Dict[str, Any]]:
    """
    “智能版”分类入口，给 router 用：

    返回:
      - final_label: 最终类别（diagram / datachart / ocrchart / photo）
      - detail: {
          "source": "clip" | "gpt" | "fallback",
          "clip_pred": ...,
          "clip_probs": {...} or None,
          "clip_error": str or None,
          "gpt_pred": ...,
          "gpt_raw": 原始返回,
        }

    参数说明：
      - use_clip: 是否尝试用 CLIP
      - gpt_fn: 你未来的小模型回调，形如  gpt_fn(prompt: str) -> str
      - prefer: 当 clip 与 gpt 都可用且结果不同，谁优先： "clip" / "gpt"
    """
    detail: Dict[str, Any] = {
        "source": "fallback",
        "clip_pred": None,
        "clip_probs": None,
        "clip_error": None,
        "gpt_pred": None,
        "gpt_raw": None,
    }

    # 1) 先尝试 CLIP（如果 use_clip=True 且本地装了）
    clip_label = None
    clip_probs: Dict[str, float] | None = None
    if use_clip:
        try:
            clip_label, clip_probs = classify_image_with_clip(image_path, device=device)
            detail["clip_pred"] = clip_label
            detail["clip_probs"] = clip_probs
        except ClipNotAvailable as e:
            detail["clip_error"] = str(e)

    # 2) 再尝试 GPT 小模型（如果传了 gpt_fn）
    gpt_label = None
    if gpt_fn is not None:
        prompt = build_gpt_prompt(caption, paragraphs)
        raw = gpt_fn(prompt)
        detail["gpt_raw"] = raw
        gpt_label = normalize_llm_label(raw)
        detail["gpt_pred"] = gpt_label

    # 3) 融合策略
    final_label = None

    if clip_label and not gpt_label:
        final_label = clip_label
        detail["source"] = "clip"
    elif gpt_label and not clip_label:
        final_label = gpt_label
        detail["source"] = "gpt"
    elif clip_label and gpt_label:
        if prefer == "gpt" and gpt_label is not None:
            final_label = gpt_label
            detail["source"] = "gpt"
        else:
            final_label = clip_label
            detail["source"] = "clip"
    else:
        # clip / gpt 都没有给有效输出 -> 最后兜底成 photo
        final_label = "photo"
        detail["source"] = "fallback"

    return final_label, detail


# ========= 兼容旧接口：classify_image =========
def classify_image(image_path: str, device: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    兼容 mm_router2 中的旧接口:
        pred_label, prob_dict = classify_image(path, device=...)
    内部直接调用纯 CLIP 版本。
    """
    return classify_image_with_clip(image_path, device=device)
