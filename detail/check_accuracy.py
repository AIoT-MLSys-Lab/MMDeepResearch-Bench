import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from dotenv import load_dotenv

# ==============================================================================
#  ENV
# ==============================================================================
load_dotenv()

# ==============================================================================
#  Gemini API
# ==============================================================================

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def get_gemini_client():
    if not genai:
        raise ImportError("google-genai not installed")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not found")
    return genai.Client(api_key=key)


def gemini_json(
    *,
    prompt: str,
    json_schema: Dict[str, Any],
    model: str,
    image_paths: Optional[List[str]],
    temperature: float,
) -> str:
    client = get_gemini_client()

    contents: List[Any] = []
    if image_paths:
        for p in image_paths:
            try:
                contents.append(Image.open(p))
            except Exception as e:
                print(f"[Warn] Cannot open image {p}: {e}")
    contents.append(prompt)

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=json_schema,
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=cfg,
        )
        return resp.text or "{}"
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "{}"


# ==============================================================================
#  Azure OpenAI API
# ==============================================================================

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


def get_azure_client():
    if not AzureOpenAI:
        raise ImportError("openai not installed")

    key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    if not (key and endpoint and api_version):
        raise ValueError("Azure OpenAI env vars missing")

    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def azure_json(
    *,
    prompt: str,
    json_schema: Dict[str, Any],
    model: str,
    temperature: float,
) -> str:
    client = get_azure_client()

    system_msg = (
        "You are a STRICT JSON generator.\n"
        "Return ONLY valid JSON exactly matching this schema:\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}\n"
        "No markdown. No explanation. No extra text."
    )

    try:
        resp = client.chat.completions.create(
            model=model,  # Azure deployment name
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
             max_completion_tokens=512,
        )
        return resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"[Azure Error] {e}")
        return "{}"


# ==============================================================================
#  Judge Router
# ==============================================================================

def judge_json(
    *,
    provider: str,
    prompt: str,
    json_schema: Dict[str, Any],
    model: str,
    temperature: float,
    image_paths: Optional[List[str]],
) -> str:
    provider = provider.lower()
    if provider == "gemini":
        return gemini_json(
            prompt=prompt,
            json_schema=json_schema,
            model=model,
            image_paths=image_paths,
            temperature=temperature,
        )
    if provider == "azure":
        return azure_json(
            prompt=prompt,
            json_schema=json_schema,
            model=model,
            temperature=temperature,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ==============================================================================
#  Utils
# ==============================================================================

def parse_json_loose(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reports(report_dir: Path) -> Dict[int, Dict[str, Any]]:
    out = {}
    for p in report_dir.glob("R*.md"):
        m = re.match(r"R(\d+)\.md", p.name)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        out[idx] = {"answer": p.read_text(encoding="utf-8"), "file": str(p)}
    return out


def resolve_image_paths(root: Path, rels: Any) -> List[str]:
    if isinstance(rels, str):
        rels = [rels]
    out = []
    if not isinstance(rels, list):
        return out
    for r in rels:
        p = Path(r)
        fp = p if p.is_absolute() else root / p
        if fp.exists():
            out.append(str(fp))
    return out


# ==============================================================================
#  Judge Schema & Prompt
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


def build_prompt(segment: str, q: str, gt: str, ans: str) -> str:
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
#  Core Eval
# ==============================================================================

def evaluate(
    *,
    quiz,
    gt,
    reports,
    image_root,
    provider,
    model,
    temperature,
    academic_cutoff,
    sleep_s,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, q in enumerate(quiz):
        if idx not in reports:
            continue

        segment = "ACADEMIC" if idx < academic_cutoff else "DAILY"
        imgs = resolve_image_paths(image_root, q.get("image_url"))
        prompt = build_prompt(
            segment,
            q.get("body", ""),
            gt[idx].get("body", ""),
            reports[idx]["answer"],
        )

        raw = judge_json(
            provider=provider,
            prompt=prompt,
            json_schema=JUDGE_SCHEMA,
            model=model,
            temperature=temperature,
            image_paths=imgs,
        )
        res = parse_json_loose(raw)

        score = int(res.get("score", 0) or 0)
        verdict = res.get("verdict", "FAIL").upper()
        if score < 6:
            verdict = "FAIL"

        print(f"[{idx:03d}] {segment} -> {verdict} ({score}/10)")

        records.append({
            "Index": idx,
            "Segment": segment,
            "Score": score,
            "Verdict": verdict,
            "Reason": res.get("reason", ""),
            "AnswerFile": reports[idx]["file"],
        })

        time.sleep(sleep_s)

    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ==============================================================================
#  Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiz_file", default="quiz.json")
    ap.add_argument("--gt_file", default="quiz_gt.json")
    ap.add_argument("--runs_root", default="../../reports_runs")
    ap.add_argument("--run_name", default="gemini-2.5-pro")
    ap.add_argument("--reports_subdir", default="reports")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--provider", choices=["gemini", "azure"], default="azure")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--academic_cutoff", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    quiz = load_json_list(args.quiz_file)
    gt = load_json_list(args.gt_file)
    report_dir = Path(args.runs_root) / args.run_name / args.reports_subdir
    reports = load_reports(report_dir)

    out = Path(args.output_dir) if args.output_dir else Path(f"evaluate_{args.run_name}")

    evaluate(
        quiz=quiz,
        gt=gt,
        reports=reports,
        image_root=Path(args.image_root),
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        academic_cutoff=args.academic_cutoff,
        sleep_s=args.sleep,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
