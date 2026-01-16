# mm_router.py (融合版：包含第二层 CLIP / Gemini 分类 + 来源文件标记)

import os
import re
import json
import io
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ======= 依赖检查与导入 =======

# 1. PDF 处理依赖
try:
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError:
    fitz = None
    Image = None

# 2. Gemini AI 依赖（用于 PDF AI 增强）
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Part
    # 尝试导入 API_KEY，如果没有则禁用增强功能
    try:
        from .api import API_KEY
        ENABLE_AI_ENHANCEMENT = True
    except ImportError:
        API_KEY = None
        ENABLE_AI_ENHANCEMENT = False
        print("⚠️ 未找到 api.py 或 API_KEY，AI 增强功能将不可用。")
except ImportError:
    genai = None
    ENABLE_AI_ENHANCEMENT = False
    print("⚠️ 未安装 google-genai，AI 增强功能将不可用。")

# ======= 模型配置 =======

MODEL_NAME = "gemini-2.0-flash"
if ENABLE_AI_ENHANCEMENT and API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


# ======= 通用数据结构 =======

@dataclass
class MMItem:
    index: int
    image_path: str                # url / 本地路径 / 占位符 image1 等
    paragraphs: List[str]          # 与图对齐的文本（上下文/说明）
    type: str                      # diagram / datachart / ocrchart / photo
    caption: str
    source_file: str               # <---【新增】所属文章/文件名 (如 "report.pdf")
    cross_page: bool = False       # pdf 跨页合并
    page: Optional[int] = None     # pdf 页码，markdown 为 None
    doc_type: str = "markdown"     # "markdown" 或 "pdf"
    missing_image: bool = False    # 文本中引用但没有真实图片时为 True


def classify_image_type_from_text(text: str) -> str:
    """根据说明文字做一层 rule-based 粗分类，作为后续 CLIP/LLM 的初始类型"""
    t = text.lower()
    # diagram
    if any(k in text for k in ["示意图", "结构图", "流程图", "架构图", "关系图"]):
        return "diagram"
    if any(k in t for k in ["diagram", "architecture", "flowchart"]):
        return "diagram"
    # datachart
    if any(k in text for k in ["图表", "折线图", "柱状图", "饼图", "统计图", "趋势图", "增长"]):
        return "datachart"
    if any(k in t for k in ["chart", "graph", "%", "trend"]):
        return "datachart"
    # ocrchart / table-like
    if any(k in text for k in ["表格", "清单", "明细表", "数据表", "截图"]):
        return "ocrchart"
    if any(k in t for k in ["table", "spreadsheet", "screenshot"]):
        return "ocrchart"
    return "photo"


# ======================
#   第二层分类器 & Gemini
#   （来自原 mm_router2.py）
# ======================

try:
    # 从 clip_classifier 里拿到：
    #   - classify_image: 纯 CLIP 四类分类
    #   - build_gpt_prompt / normalize_llm_label: 复用给 LLM（Gemini）做文字版分类
    from .clip_classifier import classify_image, build_gpt_prompt, normalize_llm_label
    HAS_CLIP = True
except Exception as e:  # 没装 open_clip 或没写 clip_classifier 时兜底
    HAS_CLIP = False
    _IMPORT_ERR = str(e)

    def classify_image(*args, **kwargs):
        raise RuntimeError(f"clip_classifier not available: {_IMPORT_ERR}")

    def build_gpt_prompt(caption: str, paragraphs: Optional[List[str]] = None) -> str:
        return ""

    def normalize_llm_label(raw: str) -> Optional[str]:
        return None

# 可选：Gemini 文本模型，用作 LLM 分类器
try:
    from .api import gemini_text
    HAS_GEMINI = True
except Exception as e:
    HAS_GEMINI = False
    _GEMINI_IMPORT_ERR = str(e)


def _run_gemini_classifier(
    caption: str,
    paragraphs: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """
    用 Gemini 做一个“文字版”的四类分类。
    """
    if not HAS_GEMINI:
        return {"label": None, "raw": None, "error": "Gemini not available"}

    prompt = build_gpt_prompt(caption=caption or "", paragraphs=paragraphs or [])
    try:
        raw = gemini_text(prompt)
    except Exception as e:
        return {"label": None, "raw": None, "error": str(e)}

    label = normalize_llm_label(raw)
    return {"label": label, "raw": raw, "error": None}


def refine_type_with_models(
    item: Dict[str, Any],
    use_clip: bool = True,
    prob_thresh: float = 0.5,
    use_gemini: bool = False,
    prefer: str = "clip",
) -> Dict[str, Any]:
    """
    综合 rule / CLIP / Gemini 三种来源，决定最终 type。
    """
    # ---- 1) 原始类型（rule-based） ----
    type_rule = (item.get("type") or "photo").lower()
    item["type_rule"] = type_rule

    # 默认先设 final
    item["type_final"] = type_rule
    item["clip_used"] = False
    item["gemini_used"] = False
    item["cls_source"] = "rule"

    caption = item.get("caption") or ""
    paragraphs = item.get("paragraphs") or []

    img_path = (item.get("image_path") or "").strip()
    has_image_file = bool(img_path) and os.path.exists(img_path)

    # 如果既不允许 CLIP 也不允许 Gemini，就直接返回 rule 结果
    if not use_clip and not use_gemini:
        return item

    # ---- 2) CLIP 预测 ----
    clip_label = None
    clip_probs: Optional[Dict[str, float]] = None
    clip_error: Optional[str] = None

    if use_clip and HAS_CLIP and has_image_file:
        try:
            clip_label, clip_probs = classify_image(img_path)
            item["clip_used"] = True
        except Exception as e:
            clip_error = str(e)

    item["clip_pred"] = clip_label
    item["clip_probs"] = clip_probs
    if clip_error:
        item["clip_error"] = clip_error

    # 判断 CLIP 预测是否“足够自信”
    clip_ok = False
    if clip_label and clip_probs and clip_label in clip_probs:
        if clip_probs[clip_label] >= prob_thresh:
            clip_ok = True

    # ---- 3) Gemini / LLM 预测（只用文本，不看图片） ----
    gpt_label = None
    gpt_raw = None
    gpt_error = None

    if use_gemini and HAS_GEMINI:
        gem_res = _run_gemini_classifier(caption=caption, paragraphs=paragraphs)
        gpt_label = gem_res.get("label")
        gpt_raw = gem_res.get("raw")
        gpt_error = gem_res.get("error")

    item["gpt_pred"] = gpt_label
    item["gpt_raw"] = gpt_raw
    if gpt_error:
        item["gpt_error"] = gpt_error

    if gpt_label:
        item["gemini_used"] = True

    # ---- 4) 综合决策 ----
    final_label = type_rule
    src = "rule"

    if prefer == "gpt":
        if gpt_label is not None:
            final_label = gpt_label
            src = "gpt"
        elif clip_ok and clip_label is not None:
            final_label = clip_label
            src = "clip"
    else:
        # prefer == "clip"
        if clip_ok and clip_label is not None:
            final_label = clip_label
            src = "clip"
        elif gpt_label is not None:
            final_label = gpt_label
            src = "gpt"

    # 兜底：确保 final_label 是我们支持的四类之一；否则退回 rule
    if final_label not in {"diagram", "datachart", "ocrchart", "photo"}:
        final_label = type_rule
        src = "rule"

    item["type_final"] = final_label
    item["cls_source"] = src
    return item


def finalize_type_and_ocr_flags(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 type_final 落实到 type 上，并根据“是否是截图”决定 OCR 标记
    """
    t_final = (
        item.get("type_final")
        or item.get("type_rule")
        or item.get("type")
        or "photo"
    ).lower()

    img_path = (item.get("image_path") or "").strip()
    has_image = bool(img_path)

    # 只要是 chart 类（datachart / ocrchart），用截图 vs 文字来区分
    if t_final in {"datachart", "ocrchart"}:
        if has_image:
            # 截图的 chart/table -> 统一走 ocrchart 流程
            item["type"] = "ocrchart"
            item["ocr_needed"] = True
            item["table_source"] = "screenshot"
        else:
            # 没有图片的 chart/table -> 统一当作文本表/结构化 chart
            item["type"] = "datachart"
            item["ocr_needed"] = False
            item["table_source"] = "text"
    else:
        # 非 chart（diagram / photo 等），保持原类型，不做 chart 层处理
        item["type"] = t_final
        item["ocr_needed"] = False
        item["table_source"] = "none"

    return item



def route_by_type(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按最终 type 分桶
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "diagram": [],
        "datachart": [],
        "ocrchart": [],
        "photo": [],
    }

    for it in items:
        t = (it.get("type") or "photo").lower()
        if t not in buckets:
            t = "photo"
        buckets[t].append(it)

    return buckets


def classify_mm_items(
    items: List[Dict[str, Any]],
    use_clip: bool = True,
    prob_thresh: float = 0.5,
    use_gemini: bool = False,
    prefer: str = "clip",
) -> List[Dict[str, Any]]:
    """
    对 mm_router 生成的 items 做统一的类型精炼
    """
    if use_clip and not HAS_CLIP:
        print("⚠️ 未能导入 clip_classifier 或 open_clip，自动退回到非 CLIP 模式。")
        use_clip = False

    if use_gemini and not HAS_GEMINI:
        print("⚠️ 未能导入 api.gemini_text 或 google-genai，Gemini 分类将被禁用。")
        use_gemini = False

    routed_items: List[Dict[str, Any]] = []
    for it in items:
        # Image-image consistency items are special: keep their type as imgpair and
        # avoid overwriting it with the 4-class router.
        if (it.get("mm_mode") == "img_img") or ((it.get("type_final") or "").lower() == "imgpair"):
            it["type_rule"] = "imgpair"
            it["type_final"] = "imgpair"
            it["cls_source"] = it.get("cls_source") or "pair"
            routed_items.append(it)
            continue

        it = refine_type_with_models(
            it,
            use_clip=use_clip,
            prob_thresh=prob_thresh,
            use_gemini=use_gemini,
            prefer=prefer,
        )
        it = finalize_type_and_ocr_flags(it)
        routed_items.append(it)

    return routed_items


# =======================
#       AI 增强模块
# =======================

def _render_page_image(pdf_path: str, page_num: int) -> Optional["Image.Image"]:
    """渲染 PDF 某一页为高清图片"""
    if not fitz or not Image:
        return None
    try:
        doc = fitz.open(pdf_path)
        # page_num 从 1 开始，fitz 索引从 0 开始
        if page_num - 1 >= len(doc):
            return None
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"      [AI] 页面渲染失败 (Page {page_num}): {e}")
        return None


def _call_gemini_for_context(page_img: "Image.Image", crop_img_path: str) -> Dict[str, Any]:
    """发送 [整页图, 切片图] 给 Gemini，提取 Caption 和 Context"""
    if not client:
        return {}

    # 读取切片图
    try:
        with open(crop_img_path, "rb") as f:
            crop_bytes = f.read()
    except Exception:
        return {}

    # 把整页图转 bytes
    buf = io.BytesIO()
    page_img.save(buf, format="JPEG")
    page_bytes = buf.getvalue()

    prompt = """
你是一个文档分析专家。我给你两张图：
1. 第一张是【PDF 完整页面】。
2. 第二张是该页面中的【插图/图表切片】。

请执行以下任务：
1. 在完整页面中定位该插图。
2. **精确提取**该插图的标题（Caption，如 "Figure 1: ..."）。
3. **分析语义**，提取正文中**专门描述或分析该插图**的段落（Paragraphs）。
   - 只提取紧密相关的部分，不要整页复制。
   - 如果图中有文字（如表格），简要概括其内容。

请返回纯 JSON 格式：
{
  "caption": "提取到的图注",
  "paragraphs": ["相关段落1", "相关段落2"],
  "type": "diagram/datachart/ocrchart/photo"
}
"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                Part.from_bytes(data=page_bytes, mime_type="image/jpeg"),
                Part.from_bytes(data=crop_bytes, mime_type="image/png"),
            ],
            config=GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"      [AI] API 调用错误: {e}")
        return {}


def enhance_pdf_items(items: List[Dict[str, Any]], pdf_path: str) -> List[Dict[str, Any]]:
    """
    遍历解析后的 items，对上下文缺失的 PDF 图片项进行 AI 增强。
    """
    if not ENABLE_AI_ENHANCEMENT:
        return items

    print(f"  ⚡ 正在启动 AI 增强 (Model: {MODEL_NAME})...")
    page_cache = {}  # 缓存页面渲染结果

    for item in items:
        # 仅处理 PDF、类型为 image 且非占位符的项
        if item.get("doc_type") != "pdf" or item.get("missing_image"):
            continue

        # 排除纯文字表格
        if item.get("type") == "ocrchart" and not item.get("image_path"):
            continue

        # 判断是否需要增强：Paragraphs 为空或内容太短
        current_paras = item.get("paragraphs", [])
        total_len = sum(len(p) for p in current_paras)
        is_weak_context = (total_len < 50)

        if is_weak_context and item.get("image_path") and os.path.exists(item["image_path"]):
            idx = item.get("index")
            pg = item.get("page")
            print(f"      -> 增强 Item {idx} (Page {pg})...", end="", flush=True)

            # 1. 获取页面图
            if pg not in page_cache:
                page_cache[pg] = _render_page_image(pdf_path, pg)

            page_img = page_cache[pg]
            if not page_img:
                print(" [跳过: 渲染失败]")
                continue

            # 2. 调用 AI
            ai_result = _call_gemini_for_context(page_img, item["image_path"])

            # 3. 更新数据
            if ai_result:
                # Caption
                raw_cap = ai_result.get("caption")
                if raw_cap:
                    item["caption"] = str(raw_cap).strip()

                # Paragraphs
                raw_paras = ai_result.get("paragraphs")
                if isinstance(raw_paras, list) and raw_paras:
                    item["paragraphs"] = [str(p).strip() for p in raw_paras if p]
                elif isinstance(raw_paras, str) and raw_paras:
                    item["paragraphs"] = [raw_paras.strip()]

                # Type (可选更新：这里的 type 也会作为 rule 层的输入之一)
                if ai_result.get("type"):
                    item["type"] = str(ai_result.get("type")).lower()

                print(" [完成]")
            else:
                print(" [无有效响应]")

            # 避免 API 速率限制
            time.sleep(1)

    return items


# =======================
#       Markdown Router
# =======================

def _is_image_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if re.search(r"!\[[^\]]*\]\([^)]+\)", line):
        return True
    if re.search(r"\[[^\]]*(图片链接|图表链接)[^\]]*\]\([^)]+\)", line):
        return True
    return False


def _extract_image_url(line: str) -> str:
    m = re.search(r"\(([^)]+)\)", line)
    return m.group(1).strip() if m else ""


def parse_markdown_blocks(md_path: str) -> List[Dict[str, Any]]:
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    blocks = []
    cur_para: List[str] = []

    def flush_para():
        nonlocal cur_para
        if cur_para:
            text = "\n".join(cur_para).strip()
            if text:
                blocks.append({"type": "paragraph", "text": text})
            cur_para = []

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            flush_para()
            continue
        if _is_image_line(stripped):
            flush_para()
            url = _extract_image_url(stripped)
            blocks.append({"type": "image", "raw": stripped, "url": url})
            continue
        cur_para.append(line)
    flush_para()
    return blocks


def _is_table_figure_block(text: str) -> bool:
    has_fig_table = bool(re.search(r"图\s*\d+.*表", text))
    has_pipe = "|" in text
    return has_fig_table and has_pipe


def _extract_figure_index_rows(blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for blk in blocks:
        if blk["type"] != "paragraph":
            continue
        txt = blk["text"]
        if "| 图号 |" not in txt or "场景描述" not in txt:
            continue
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for line in lines:
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 3:
                continue
            fig_id = parts[0]
            if fig_id in ("图号", ":---"):
                continue
            if re.fullmatch(r"图\s*\d+", fig_id):
                rows.append(
                    {
                        "fig_id": fig_id.replace(" ", ""),
                        "scene": parts[1],
                        "tech": parts[2],
                    }
                )
    return rows


def _resolve_figure_image_path(fig_id: str, figure_image_map: Optional[Dict[str, str]]) -> str:
    if not figure_image_map:
        return ""
    num_match = re.findall(r"\d+", fig_id)
    num = num_match[0] if num_match else fig_id
    candidates = [fig_id, f"图{num}", num, f"image{num}", f"图片{num}"]
    for key in candidates:
        if key in figure_image_map:
            return figure_image_map[key]
    return ""


def build_mm_items_from_markdown(
    blocks: List[Dict[str, Any]],
    figure_image_map: Optional[Dict[str, str]] = None,
    source_file: str = "",  # <---【新增】
) -> List[MMItem]:
    items: List[MMItem] = []
    idx_counter = 1

    # 0) 图1/图2 表格定义的图
    fig_rows = _extract_figure_index_rows(blocks)
    for row in fig_rows:
        fig_id = row["fig_id"]
        img_path = _resolve_figure_image_path(fig_id, figure_image_map)
        missing_image = False if img_path else True
        if not img_path:
            num_match = re.findall(r"\d+", fig_id)
            num = num_match[0] if num_match else fig_id
            img_path = f"image{num}"

        scene, tech = row["scene"], row["tech"]
        caption = f"{fig_id}：{scene}"
        paras = [f"场景描述：{scene}", f"摄影技法特征：{tech}"]

        # rule 层先给一个粗 type
        type_text = caption + "\n" + "\n".join(paras)
        t = classify_image_type_from_text(type_text)

        items.append(
            MMItem(
                index=idx_counter,
                image_path=img_path,
                paragraphs=paras,
                type=t,
                caption=caption,
                missing_image=missing_image,
                source_file=source_file,  # <---【填入】
            )
        )
        idx_counter += 1

    # 1) Image Blocks
    for i, blk in enumerate(blocks):
        if blk["type"] != "image":
            continue
        img_url = blk["url"]
        raw = blk.get("raw", "")
        caption = ""
        if i > 0 and blocks[i - 1]["type"] == "paragraph":
            prev_text = blocks[i - 1]["text"].strip()
            if re.search(r"图\s*\d+", prev_text):
                caption = prev_text
        if not caption:
            caption = raw

        paras: List[str] = []
        # 向前找一段
        for back in range(i - 1, -1, -1):
            if blocks[back]["type"] == "paragraph":
                pt = blocks[back]["text"].strip()
                if pt and pt != caption:
                    paras.insert(0, pt)
                    break
        # 向后找一段
        for fwd in range(i + 1, len(blocks)):
            if blocks[fwd]["type"] == "paragraph":
                pt = blocks[fwd]["text"].strip()
                if pt:
                    paras.append(pt)
                    break

        type_text = caption + "\n" + "\n".join(paras) + "\n" + raw
        t = classify_image_type_from_text(type_text)

        items.append(
            MMItem(
                index=idx_counter,
                image_path=img_url,
                paragraphs=paras,
                type=t,
                caption=caption,
                source_file=source_file,  # <---【填入】
            )
        )
        idx_counter += 1

    # 2) Table Figures（纯文本表格）
    for blk in blocks:
        if blk["type"] == "paragraph" and _is_table_figure_block(blk["text"]):
            text = blk["text"]
            lines = [l for l in text.splitlines() if l.strip()]
            first_line = lines[0] if lines else text.strip()
            m = re.search(r"图\s*\d+[^*\n]*表", first_line)
            cap = m.group(0) if m else first_line
            items.append(
                MMItem(
                    index=idx_counter,
                    image_path="",
                    paragraphs=[text],
                    type="ocrchart",
                    caption=cap,
                    source_file=source_file,  # <---【填入】
                )
            )
            idx_counter += 1
    return items


def route_markdown(
    md_path: str,
    figure_image_map: Optional[Dict[str, str]] = None,
    source_file: str = "",  # <---【新增】
) -> List[Dict[str, Any]]:
    blocks = parse_markdown_blocks(md_path)
    mm_items = build_mm_items_from_markdown(blocks, figure_image_map, source_file=source_file)
    result: List[Dict[str, Any]] = []
    for item in mm_items:
        result.append(dict(item.__dict__))
    return result


# =======================
#        PDF Router
# =======================

def merge_images_vertically(img1: "Image.Image", img2: "Image.Image") -> "Image.Image":
    w = max(img1.width, img2.width)
    h = img1.height + img2.height
    merged = Image.new("RGB", (w, h), (255, 255, 255))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (0, img1.height))
    return merged


def extract_and_merge_clean(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    if not fitz or not Image:
        raise ImportError("需要 PyMuPDF 和 Pillow")

    doc = fitz.open(pdf_path)
    results: List[Dict[str, Any]] = []
    temp_paths: List[str] = []

    prev_img, prev_path, prev_bottom, prev_page_index = None, None, None, None
    os.makedirs(output_dir, exist_ok=True)

    for i, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        img_list = page.get_images(full=True)

        for j, img in enumerate(img_list):
            xref = img[0]

            # ① 提取图片 + 修复模式
            try:
                base_img = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_img["image"]))

                # 🔧 关键修复
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
            except Exception:
                continue

            # ② 计算 bbox，找上下文字块
            bbox = page.get_image_bbox(img)

            nearest_text, min_dist = "", 9999.0
            for blk in text_blocks:
                x0, y0, x1, y1, text, *_ = blk
                if y1 < bbox.y0 and (bbox.y0 - y1) < min_dist:
                    min_dist = bbox.y0 - y1
                    nearest_text = str(text).strip()

            cap_match = re.findall(r"(图[\d]+[：: ]?.{0,40})", nearest_text)
            cap = cap_match[0] if cap_match else nearest_text[:60]
            context = nearest_text

            # ③ 保存当前图片
            img_path = os.path.join(output_dir, f"page{i+1}_img{j+1}.png")
            image.save(img_path)

            type_text = (cap or "") + "\n" + (context or "")
            t = classify_image_type_from_text(type_text)

            cur_record = {
                "kind": "image",
                "page": i + 1,
                "image": img_path,
                "caption": cap,
                "context": context,
                "type": t,
                "cross_page": False,
                "pos": float(bbox.y0),
            }

            # ④ 跨页合并检测
            if (
                prev_img
                and prev_page_index == i - 1
                and prev_bottom is not None
                and prev_bottom > page.rect.height * 0.9
                and bbox.y0 < page.rect.height * 0.15
            ):
                merged_img = merge_images_vertically(prev_img, image)
                merged_path = os.path.join(output_dir, f"merged_page{i}_img{j}.png")
                merged_img.save(merged_path)

                if os.path.exists(prev_path):
                    os.remove(prev_path)
                    temp_paths.append(prev_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    temp_paths.append(img_path)

                # 把之前那条替换成 merged 版本
                results = [r for r in results if r["image"] != prev_path]
                results.append(
                    {
                        "kind": "image",
                        "page": i + 1,
                        "image": merged_path,
                        "caption": cap,
                        "context": context,
                        "type": t,
                        "cross_page": True,
                        "pos": 0.0,
                    }
                )

                prev_img, prev_path, prev_bottom, prev_page_index = None, None, None, None
                continue

            # 正常情况：直接记录当前图片
            results.append(cur_record)
            prev_img, prev_path, prev_bottom, prev_page_index = (
                image,
                img_path,
                bbox.y1,
                i,
            )

    # 去掉被合并后删掉的临时图片
    final_results: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        if r["image"] not in temp_paths and r["image"] not in seen:
            seen.add(r["image"])
            final_results.append(r)

    return final_results



def extract_pdf_tables(pdf_path: str) -> List[Dict[str, Any]]:
    if not fitz:
        return []
    doc = fitz.open(pdf_path)
    table_records: List[Dict[str, Any]] = []
    for page_index, page in enumerate(doc):
        text = page.get_text("text")
        if "Table" not in text:
            continue
        cursor = 0
        length = len(text)
        while True:
            start = text.find("Table", cursor)
            if start == -1:
                break
            next_table = text.find("Table", start + 5)
            if next_table == -1:
                next_table = length
            src_idx = text.find("来源", start)
            end_candidates = []
            if src_idx != -1 and src_idx < next_table:
                line_end = text.find("\n", src_idx)
                end_candidates.append(line_end if line_end != -1 else length)
            end_candidates.append(next_table)
            end = min(e for e in end_candidates if e > start)
            table_text = text[start:end].strip()
            cursor = end
            if not table_text:
                continue
            lines = [l.strip() for l in table_text.splitlines() if l.strip()]
            cap_line = lines[0] if lines else "Table"

            # 简单估算位置
            pos_y = 0.0
            try:
                rects = page.search_for(cap_line[:30])
                if rects:
                    pos_y = float(rects[0].y0)
            except Exception:
                pass

            table_records.append(
                {
                    "kind": "table",
                    "page": page_index + 1,
                    "caption": cap_line,
                    "text": table_text,
                    "pos": pos_y,
                }
            )
    return table_records


def build_mm_items_from_pdf(
    pdf_path: str, 
    output_dir: str,
    source_file: str = ""  # <---【新增】
) -> List[MMItem]:
    image_records = extract_and_merge_clean(pdf_path, output_dir)
    table_records = extract_pdf_tables(pdf_path)

    combined: List[Dict[str, Any]] = []
    for r in image_records:
        combined.append(
            {
                "kind": "image",
                "page": r["page"],
                "pos": r.get("pos", 0.0),
                "image": r["image"],
                "caption": r.get("caption", ""),
                "context": r.get("context", ""),
                "type": r.get("type", "photo"),
                "cross_page": r.get("cross_page", False),
            }
        )
    for t in table_records:
        combined.append(
            {
                "kind": "table",
                "page": t["page"],
                "pos": t.get("pos", 0.0),
                "image": "",
                "caption": t.get("caption", ""),
                "context": t.get("text", ""),
                "type": "ocrchart",
                "cross_page": False,
            }
        )
    combined.sort(key=lambda r: (r["page"], r.get("pos", 0.0)))

    items: List[MMItem] = []
    idx = 1
    for r in combined:
        para = r.get("context") or r.get("caption") or ""
        items.append(
            MMItem(
                index=idx,
                image_path=r["image"],
                paragraphs=[para],
                type=r["type"],
                caption=r["caption"],
                cross_page=r["cross_page"],
                page=r["page"],
                doc_type="pdf",
                missing_image=False,
                source_file=source_file,  # <---【填入】
            )
        )
        idx += 1
    return items


def route_pdf(
    pdf_path: str, 
    image_output_dir: str = "split_output_final",
    source_file: str = ""  # <---【新增】
) -> List[Dict[str, Any]]:
    mm_items = build_mm_items_from_pdf(pdf_path, image_output_dir, source_file=source_file)
    result: List[Dict[str, Any]] = []
    for item in mm_items:
        result.append(dict(item.__dict__))
    return result


# =======================
#       统一入口 Router
# =======================

def route_document(
    path: str,
    figure_image_map: Optional[Dict[str, str]] = None,
    pdf_image_output_dir: str = "split_output_final",
    source_file: str = "",  # <---【新增】
) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".md", ".markdown"]:
        return route_markdown(path, figure_image_map=figure_image_map, source_file=source_file)
    elif ext == ".pdf":
        return route_pdf(path, image_output_dir=pdf_image_output_dir, source_file=source_file)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


if __name__ == "__main__":
    import sys

    # 1. 设置文件列表
    if len(sys.argv) > 1:
        files_to_process = sys.argv[1:]
    else:
        files_to_process = ["example.pdf", "example2.pdf"]

    all_results: List[Dict[str, Any]] = []

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过 (文件不存在): {file_path}")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        # 提取文件名作为 ID (例如: "example.pdf")
        doc_id = os.path.basename(file_path)
        stem = os.path.splitext(doc_id)[0]
        
        print(f"➡️ 正在处理: {file_path} (ID: {doc_id}) ...")

        try:
            items: List[Dict[str, Any]] = []
            if ext == ".pdf":
                # 定义 PDF 图片输出目录
                pdf_img_dir = os.path.join("split_output_final", stem)

                # 1. 基础解析（传入 source_file=doc_id）
                items = route_document(file_path, pdf_image_output_dir=pdf_img_dir, source_file=doc_id)

                # 2. AI 增强
                if ENABLE_AI_ENHANCEMENT and items:
                    items = enhance_pdf_items(items, file_path)
            else:
                # Markdown 处理（传入 source_file=doc_id）
                items = route_document(file_path, source_file=doc_id)

            # 3. CLIP / Gemini 二阶段类型精炼
            if items:
                items = classify_mm_items(
                    items,
                    use_clip=False,      # 有 clip_classifier 时启用 CLIP
                    prob_thresh=0.5,
                    use_gemini=True,   # 如果想用 Gemini 文本分类，这里改成 True
                    prefer="clip",
                )

            all_results.extend(items)
            print(f"   ✅ 成功提取 {len(items)} 条数据")

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback

            traceback.print_exc()

    # 4. 重新生成连续 Index（跨文件重新编号）
    for i, item in enumerate(all_results):
        item["index"] = i + 1

    # 5. 保存最终 JSON
    output_json_path = "mm_router_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 6. 额外输出按类型分桶的结果
    routed_dir = "mm_router_routed"
    os.makedirs(routed_dir, exist_ok=True)

    # all_routed.json
    all_routed_path = os.path.join(routed_dir, "all_routed.json")
    with open(all_routed_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # diagram/datachart/ocrchart/photo 四个文件
    buckets = route_by_type(all_results)
    for t, arr in buckets.items():
        out_path = os.path.join(routed_dir, f"{t}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
        print(f"{t:9s}: {len(arr):3d} items -> {out_path}")

    print("=" * 40)
    print(f"总样本数: {len(all_results)}")
    print(f"📁 汇总文件: {output_json_path}")
    print(f"📁 二阶段路由结果: {all_routed_path} 以及 {routed_dir}/[diagram|datachart|ocrchart|photo].json")