# sep.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict
from docx import Document


# ===============================
# 正则定义（核心）
# ===============================

# 真正的题号：A1 ~ K99（不认 I. / A. 这种）
QUESTION_ID_RE = re.compile(r"\b([A-K])\s*([1-9]\d?)\b")

# 分组标题（必须排除）
SECTION_HEADER_RE = re.compile(r"^[A-K]\s*[\.、]")

# ===============================
# 数据结构
# ===============================

class Question:
    def __init__(self, qid: str):
        self.qid = qid
        self.title = ""
        self.body_lines: List[str] = []

    def to_json(self) -> Dict:
        body = "\n".join(self.body_lines).strip()
        title = self.title.strip() if self.title else body[:20]

        return {
            "qid": self.qid,
            "caption": f"{self.qid}: {title}",
            "body": body,
            "image_url": [],
            "tags": [],
            "language": "zh",
            "difficulty": infer_difficulty(body),
        }


# ===============================
# 难度推断（可随时改）
# ===============================

def infer_difficulty(text: str) -> str:
    t = text.lower()
    if "hard" in t or "复杂" in t or "跨" in t:
        return "hard"
    if "easy" in t or "解释" in t or "说明" in t:
        return "easy"
    return "medium"


# ===============================
# 主解析函数
# ===============================

def parse_docx(docx_path: Path) -> List[Question]:
    doc = Document(docx_path)

    questions: List[Question] = []
    current: Question | None = None
    seen_qids = set()

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        # ---- 1. 跳过分组标题 ----
        if SECTION_HEADER_RE.match(text):
            continue

        # ---- 2. 搜索题号（Anywhere）----
        m = QUESTION_ID_RE.search(text)
        if m:
            letter, num = m.group(1), m.group(2)
            qid = f"{letter}{num}"

            # 防止重复触发（Word 里偶尔会重复）
            if qid in seen_qids:
                if current:
                    current.body_lines.append(text)
                continue

            seen_qids.add(qid)

            # 收尾上一题
            if current:
                questions.append(current)

            current = Question(qid)
            current.title = text
            continue

        # ---- 3. 普通正文，拼接到当前题 ----
        if current:
            current.body_lines.append(text)

    # ---- 收尾最后一题 ----
    if current:
        questions.append(current)

    return questions


# ===============================
# CLI 入口
# ===============================

def main():
    docx_path = Path("question set.docx")
    out_path = Path("quiz.json")

    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    questions = parse_docx(docx_path)

    print(f"解析得到题目数量: {len(questions)}")
    for q in questions:
        print(q.qid, q.title[:20])

    if len(questions) != 40:
        print("⚠️ 警告：题目数量不是 40，请检查 Word 结构")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [q.to_json() for q in questions],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
