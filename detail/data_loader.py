import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from .models import Task

def determine_report_type(t: Task) -> str:
    # (保持原有逻辑不变)
    meta = t.meta or {}
    if (meta.get("set") or "").lower() == "research":
        diff = (meta.get("difficulty") or "").strip().lower()
        if diff in {"easy", "hard", "complex"}:
            return f"research_{diff}"
        return "research"
    return "daily_medium"

def format_task_for_llm(t: Task) -> str:
    # (保持原有逻辑不变)
    parts: List[str] = []
    if (t.title or "").strip():
        parts.append(t.title.strip())
    if (t.text or "").strip():
        parts.append(t.text.strip())
    
    if t.images:
        parts.append(
            "[Images]\n"
            + "\n".join(f"- {p}" for p in t.images)
            + "\n\nNOTE: If images are local, use them as evidence targets."
        )
    return "\n\n".join(parts).strip()

def _load_json_list(path: Path) -> List[Dict]:
    """通用加载器：支持 .json (List) 和 .jsonl (Lines)"""
    items = []
    if not path.exists():
        print(f"❌ File not found: {path}")
        return []
    
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: items.append(json.loads(line))
                    except: pass
    else:
        try: items = json.loads(path.read_text(encoding="utf-8"))
        except: pass
    return items

def load_research_tasks(json_path: str, image_root: str, gt_path: Optional[str] = None) -> List[Task]:
    """
    加载 Task，并可选地从 gt_path 加载 Ground Truth 进行合并
    """
    p = Path(json_path) if json_path else None
    if not p or not p.exists():
        return []

    root = Path(image_root).resolve() if image_root else p.parent.resolve()
    
    # 1. 加载题目
    raw_items = _load_json_list(p)
    print(f">>> Loaded {len(raw_items)} questions from {p.name}")

    # 2. 加载 GT (如果有)
    gt_map = {} # 索引 -> GT 文本
    if gt_path:
        gp = Path(gt_path)
        if gp.exists():
            gt_items = _load_json_list(gp)
            print(f">>> Loaded {len(gt_items)} answers from {gp.name}")
            # 建立映射。这里假设 jsonl 行号一一对应（最常见），或者用 id 对应
            for idx, item in enumerate(gt_items):
                # 尝试获取 id
                qid = str(item.get("qid") or item.get("id") or idx).strip()
                ans = item.get("body") or item.get("text") or item.get("answer") or item.get("gt") or ""
                # 同时存 id 和 index，双重保障
                gt_map[str(idx)] = ans
                gt_map[qid] = ans
        else:
            print(f"⚠️ GT path provided but file not found: {gp}")

    if not raw_items:
        return []

    tasks: List[Task] = []
    seen: Dict[str, int] = {}

    for i, obj in enumerate(raw_items):
        if not isinstance(obj, dict): continue
        
        # 基础字段
        caption = (obj.get("caption") or obj.get("title") or "").strip()
        body = (obj.get("body") or obj.get("text") or "").strip()
        
        # 图片
        imgs = obj.get("image_url") or obj.get("images") or []
        if isinstance(imgs, str): imgs = [imgs]
        resolved_imgs = []
        for rel in imgs:
            if not rel: continue
            rel = str(rel).strip()
            if re.match(r"^https?://", rel, re.IGNORECASE):
                resolved_imgs.append(rel)
            else:
                rel = rel.replace("\\", "/")
                resolved_imgs.append(str((root / rel).resolve()))

        # ID
        raw_id = str(obj.get("qid") or obj.get("id") or "").strip()
        if raw_id and raw_id.isdigit(): raw_id = f"Q{raw_id}"
        if not raw_id: raw_id = f"Q{i}" # Fallback
        
        # 处理重名
        qid = raw_id
        if qid in seen:
            seen[qid] += 1
            qid = f"{qid}_{seen[qid]}"
        else:
            seen[qid] = 0

        # --- 获取 GT ---
        # 优先用 index 匹配 (对应 jsonl 顺序)，其次用 ID
        ground_truth = gt_map.get(str(i)) or gt_map.get(raw_id) or ""
        # ----------------
        
        meta = {
            "set": "research",
            "difficulty": (obj.get("difficulty") or "").strip().lower() or None,
            "source": "jsonl"
        }

        tasks.append(Task(
            qid=qid,
            title=caption or "Untitled",
            text=body or caption or "",
            images=resolved_imgs,
            meta=meta,
            ground_truth=ground_truth # 塞进去！
        ))

    return tasks