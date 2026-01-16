import os
import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

# Try imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

# ==============================================================================
# 1. 完全复刻的 Prompt 构建器
# ==============================================================================
JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "INTEGER"},
        "reason": {"type": "STRING"},
        "verdict": {"type": "STRING", "enum": ["PASS", "FAIL"]},
    },
    "required": ["score", "reason", "verdict"],
}

def build_judge_prompt(segment: str, q: str, gt: str, ans: str) -> str:
    # --- 这里完全复制你提供的脚本内容 ---
    return f"""
You are a STRICT QA Judge.
Segment = {segment}

### QUESTION
{q}

### GROUND TRUTH
{gt}

### MODEL ANSWER
{ans}

Rules:
- Any wrong visual identity => FAIL
- False presence => FAIL
- Missing details allowed only if no wrong IDs
- score < 6 MUST be FAIL

Return JSON only.
""".strip()

# ==============================================================================
# 2. Providers (Gemini & Azure)
# ==============================================================================
def _gemini_judge(prompt: str, model: str, images: List[str], temp: float) -> str:
    if not genai: return "{}"
    key = os.environ.get("GEMINI_API_KEY")
    if not key: return "{}"
    client = genai.Client(api_key=key)

    contents = []
    # Gemini 支持看图判卷
    if images:
        for p in images:
            try:
                path_obj = Path(p)
                if path_obj.exists():
                    contents.append(Image.open(path_obj))
            except: pass
    contents.append(prompt)

    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temp,
                response_mime_type="application/json",
                response_schema=JUDGE_SCHEMA
            )
        )
        return resp.text or "{}"
    except Exception as e:
        print(f"[ACC Judge Error] Gemini: {e}")
        return "{}"

def _azure_judge(prompt: str, model: str, temp: float) -> str:
    if not AzureOpenAI: return "{}"
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    end = os.environ.get("AZURE_OPENAI_ENDPOINT")
    ver = os.environ.get("AZURE_OPENAI_API_VERSION")
    if not (key and end): return "{}"
    
    client = AzureOpenAI(api_key=key, azure_endpoint=end, api_version=ver)
    
    # 强制 System Prompt 保持一致
    system_msg = (
        "You are a STRICT JSON generator.\n"
        "Return ONLY valid JSON exactly matching this schema:\n"
        f"{json.dumps(JUDGE_SCHEMA, ensure_ascii=False)}\n"
        "No markdown. No explanation. No extra text."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"[ACC Judge Error] Azure: {e}")
        return "{}"

# ==============================================================================
# 3. Main Entry
# ==============================================================================
def run_accuracy_check(
    provider: str,
    model: str,
    segment: str,   # [新增] 传入 segment
    question: str,
    gt: str,
    report: str,
    images: List[str],
    temperature: float = 0.1 # 默认跟脚本保持一致 0.1
) -> Dict[str, Any]:
    
    if not gt or not gt.strip():
        return {"score": 0, "verdict": "NO_GT", "reason": "No ground truth provided"}

    # 构建一模一样的 Prompt
    prompt = build_judge_prompt(segment, question, gt, report)
    
    p = provider.lower()
    raw = "{}"
    
    if "gemini" in p:
        raw = _gemini_judge(prompt, model, images, temperature)
    elif "azure" in p or "openai" in p:
        raw = _azure_judge(prompt, model, temperature)
    else:
        raw = _azure_judge(prompt, model, temperature)

    # Parse JSON
    try:
        data = json.loads(raw)
        return data
    except:
        # Loose Parse
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
        return {"score": 0, "verdict": "ERROR", "reason": "JSON Parse Error"}