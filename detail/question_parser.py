# question_parser.py
"""Load benchmark question sets.

Supports:
- Daily questions from .docx (legacy)
- Research questions from .json + local images (new)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

try:
    from docx import Document
except ImportError:
    Document = None


@dataclass
class Question:
    qid: str
    title: str
    text: str

    # Multimodal (research) support
    images: List[str] = field(default_factory=list)

    # Metadata (optional)
    set_name: str = "daily"  # "daily" or "research"
    tags: List[str] = field(default_factory=list)
    language: Optional[str] = None
    difficulty: Optional[str] = None


_ID_PATTERN = re.compile(r"^([A-Z]\d+)\s*(.*)$")


def _clean_title(raw: str) -> str:
    t = (raw or "").strip()
    t = t.strip("()（） ")
    t = re.sub(r"\s+", " ", t)
    return t or "Untitled"


def load_questions_from_docx(path: Path) -> List[Question]:
    if Document is None:
        raise ImportError("python-docx is not installed. Run `pip install python-docx`.")
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    doc = Document(str(path))
    questions: List[Question] = []

    current_qid: Optional[str] = None
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for para in doc.paragraphs:
        line = (para.text or "").strip()
        if not line:
            continue

        m = _ID_PATTERN.match(line)
        if m:
            if current_qid is not None:
                questions.append(
                    Question(
                        qid=current_qid,
                        title=current_title or "Untitled",
                        text="\n".join(current_lines).strip(),
                        set_name="daily",
                    )
                )
            current_qid = m.group(1)
            current_title = _clean_title(m.group(2))
            current_lines = []
            continue

        if current_qid is not None:
            current_lines.append(line)

    if current_qid is not None:
        questions.append(
            Question(
                qid=current_qid,
                title=current_title or "Untitled",
                text="\n".join(current_lines).strip(),
                set_name="daily",
            )
        )

    return questions


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def load_questions_from_research_json(json_path: Path, image_root: Path) -> List[Question]:
    """Load the research question set.

    Expected schema: a JSON list, each item like:
      {
        "caption": "...",
        "body": "...",
        "image_url": ["image/Q0I0.jpg", ...],
        "tags": ["HCS"],
        "language": "zh",
        "difficulty": "easy"
      }
    """
    if not json_path.exists():
        return []

    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        # allow {"questions":[...]}
        obj = obj.get("questions") or obj.get("items") or obj.get("data") or []
    if not isinstance(obj, list):
        return []

    qs: List[Question] = []
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            continue
        caption = str(it.get("caption") or "").strip()
        body = str(it.get("body") or "").strip()
        tags = [str(x) for x in _ensure_list(it.get("tags")) if str(x).strip()]
        language = str(it.get("language") or "").strip() or None
        difficulty = str(it.get("difficulty") or "").strip().lower() or None

        # Resolve images
        img_list = [str(x) for x in _ensure_list(it.get("image_url")) if str(x).strip()]
        images: List[str] = []
        for u in img_list:
            if u.startswith("http://") or u.startswith("https://") or u.startswith("file://"):
                images.append(u)
            else:
                images.append(str((image_root / u).resolve()))

        qid = f"R{i+1:03d}"
        title = caption or f"Research {qid}"
        if difficulty:
            title = f"{title} (Research {difficulty})"

        # Put minimal image manifest into the question text (the model also receives images via image_paths)
        img_manifest_lines = []
        for j, p in enumerate(images):
            img_manifest_lines.append(f"- Image {j}: {p}")
        img_manifest = "\n".join(img_manifest_lines)

        text = f"{caption}\n\n{body}".strip()
        if img_manifest:
            text += "\n\n[Local Images]\n" + img_manifest

        qs.append(
            Question(
                qid=qid,
                title=title,
                text=text,
                images=images,
                set_name="research",
                tags=tags,
                language=language,
                difficulty=difficulty,
            )
        )

    return qs
