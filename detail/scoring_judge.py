from __future__ import annotations

import json
import re
from typing import Any, Dict

from .api import llm_json
from .config import LLM
from .scoring_math import clamp01


def _json_loads_lenient(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s2 = s.strip()
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _schema_call(prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int) -> Dict[str, Any]:
    s = llm_json(
        prompt=prompt,
        json_schema=schema,
        model=getattr(LLM, "judge_model", "gemini-2.5-pro"),
        provider=getattr(LLM, "judge_provider", "gemini"),
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return _json_loads_lenient(s)


def judge_task_controller(question_text: str, report_type: str) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "task_family": {"type": "string"},
            "v_general": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "v_evidence": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "lambda_final": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["task_family", "v_general", "v_evidence", "lambda_final", "rationale"],
    }

    prompt = f"""
You are a scoring controller for a Deep Research benchmark.

Task:
{question_text}

Report type hint:
{report_type}

Return strict JSON:
{{
  "task_family": "analysis|survey|listing|howto|fact_check|news|other",
  "v_general": [vR, vI, vC, vS],
  "v_evidence": [vCon, vCov, vFid, vDiv],
  "lambda_final": 0.55,
  "rationale": "one short sentence"
}}

Guidance:
- Daily/news: emphasize readability & structure; lambda_final in [0.50, 0.60].
- Research/survey: emphasize insight/coherence and evidence fidelity; lambda_final around 0.35~0.45.
- Fact_check: emphasize evidence consistency strongly (Con/Fid/Cov).
"""
    obj = _schema_call(prompt, schema, temperature=0.2, max_tokens=getattr(LLM, "max_judge_tokens", 4096))
    if not obj:
        obj = {
            "task_family": "other",
            "v_general": [0.0, 0.0, 0.0, 0.0],
            "v_evidence": [0.0, 0.0, 0.0, 0.0],
            "lambda_final": 0.55,
            "rationale": "fallback",
        }

    obj["v_general"] = [float(x) for x in (obj.get("v_general") + [0, 0, 0, 0])[:4]]
    obj["v_evidence"] = [float(x) for x in (obj.get("v_evidence") + [0, 0, 0, 0])[:4]]
    obj["lambda_final"] = clamp01(float(obj.get("lambda_final", 0.55)))
    return obj


def judge_general_features(question_text: str, report_text: str, rules_text: str, report_type: str) -> Dict[str, Any]:
    sections_schema = {
        "type": "object",
        "properties": {
            "Executive Summary": {"type": "boolean"},
            "Background": {"type": "boolean"},
            "Methodology": {"type": "boolean"},
            "Analysis & Evidence": {"type": "boolean"},
            "Visual Integration": {"type": "boolean"},
            "Conclusion": {"type": "boolean"},
        },
        "required": [
            "Executive Summary",
            "Background",
            "Methodology",
            "Analysis & Evidence",
            "Visual Integration",
            "Conclusion",
        ],
    }

    schema = {
        "type": "object",
        "properties": {
            "E": {"type": "number"},
            "p_analysis": {"type": "number"},
            "rho_connective": {"type": "number"},
            "rho_synthesis": {"type": "number"},
            "U": {"type": "number"},
            "O": {"type": "number"},
            "T": {"type": "number"},
            "kappa": {"type": "number"},
            "theta": {"type": "number"},
            "sections": sections_schema,
            "llm_R": {"type": "number"},
            "llm_I": {"type": "number"},
            "llm_C": {"type": "number"},
            "llm_S": {"type": "number"},
            "notes": {"type": "string"},
        },
        "required": ["E", "p_analysis", "rho_connective", "rho_synthesis", "U", "O", "T", "kappa", "theta", "sections", "notes"],
    }

    prompt = f"""
You are a judge for General Metrics in a Deep Research benchmark.

Rules (excerpt):
{rules_text[:1400]}

Task:
{question_text}

Report (truncated):
{report_text[:14000]}

Return strict JSON (all numbers in [0,1]):
{{
  "E": 0.05,
  "p_analysis": 0.40,
  "rho_connective": 0.60,
  "rho_synthesis": 0.50,
  "U": 0.70,
  "O": 0.65,
  "T": 0.75,
  "kappa": 0.02,
  "theta": 0.80,
  "sections": {{
    "Executive Summary": true,
    "Background": true,
    "Methodology": false,
    "Analysis & Evidence": true,
    "Visual Integration": false,
    "Conclusion": true
  }},
  "llm_R": 0.70,
  "llm_I": 0.60,
  "llm_C": 0.65,
  "llm_S": 0.70,
  "notes": "one short sentence"
}}

Report type hint: {report_type}
"""
    obj = _schema_call(prompt, schema, temperature=0.2, max_tokens=getattr(LLM, "max_judge_tokens", 4096))
    if not obj:
        obj = {
            "E": 0.08, "p_analysis": 0.35, "rho_connective": 0.45, "rho_synthesis": 0.35,
            "U": 0.55, "O": 0.50, "T": 0.55, "kappa": 0.02, "theta": 0.65,
            "sections": {
                "Executive Summary": False,
                "Background": False,
                "Methodology": False,
                "Analysis & Evidence": False,
                "Visual Integration": False,
                "Conclusion": False,
            },
            "llm_R": 0.60, "llm_I": 0.55, "llm_C": 0.55, "llm_S": 0.55,
            "notes": "fallback",
        }

    for k in ["E", "p_analysis", "rho_connective", "rho_synthesis", "U", "O", "T", "kappa", "theta", "llm_R", "llm_I", "llm_C", "llm_S"]:
        if k in obj:
            obj[k] = clamp01(obj.get(k, 0.0))

    if not isinstance(obj.get("sections"), dict):
        obj["sections"] = sections_schema["properties"].copy()

    return obj


def judge_evidence_pair(claim: str, url: str, excerpt: str, rules_text: str) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "support_strength": {"type": "number"},
            "hallucination_prob": {"type": "number"},
            "inversion_prob": {"type": "number"},
            "chain_sound_prob": {"type": "number"},
            "source_category": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["support_strength", "hallucination_prob", "inversion_prob", "chain_sound_prob", "source_category", "why"],
    }

    prompt = f"""
You are an Evidence Judge for a Deep Research benchmark.

Rules (excerpt):
{rules_text[:1400]}

Claim:
{claim}

URL:
{url}

Evidence excerpt:
{excerpt[:9000]}

Return strict JSON:
{{
  "support_strength": 0.0,
  "hallucination_prob": 0.0,
  "inversion_prob": 0.0,
  "chain_sound_prob": 0.0,
  "source_category": "authoritative|acceptable|prohibited|unknown",
  "why": "one short sentence"
}}
"""
    obj = _schema_call(prompt, schema, temperature=0.1, max_tokens=getattr(LLM, "max_judge_tokens", 4096))
    if not obj:
        obj = {
            "support_strength": 0.35,
            "hallucination_prob": 0.25,
            "inversion_prob": 0.10,
            "chain_sound_prob": 0.45,
            "source_category": "unknown",
            "why": "fallback",
        }

    obj["support_strength"] = clamp01(obj.get("support_strength", 0.0))
    obj["hallucination_prob"] = clamp01(obj.get("hallucination_prob", 0.0))
    obj["inversion_prob"] = clamp01(obj.get("inversion_prob", 0.0))
    obj["chain_sound_prob"] = clamp01(obj.get("chain_sound_prob", 0.0))

    cat = str(obj.get("source_category", "unknown")).lower()
    if cat not in {"authoritative", "acceptable", "prohibited", "unknown"}:
        cat = "unknown"
    obj["source_category"] = cat
    return obj
