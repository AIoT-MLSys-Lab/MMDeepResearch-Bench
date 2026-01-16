from __future__ import annotations

import math
import re
from typing import Any, Dict, Tuple

from .scoring_math import clamp01, harmonic_mean, logistic, safe_div, softmax
from .scoring_text import (
    extract_markdown_headings, tokenize_words,
    ttr, title_density, long_sentence_ratio
)
from .scoring_judge import judge_general_features, judge_task_controller


def _alpha_formula(report_type: str) -> float:
    # daily: smaller alpha => more LLM; research: larger alpha => more formula stability
    if report_type == "research":
        return 0.60
    if report_type == "daily_hard":
        return 0.40
    if report_type == "daily_easy":
        return 0.30
    return 0.35


def _compute_structural_completeness(report_text: str, section_flags: Dict[str, bool], theta: float, report_type: str) -> Tuple[float, Dict[str, Any]]:
    # Daily tasks: looser required set (avoid product=0 too easily)
    if report_type == "research":
        required = [
            "Executive Summary",
            "Background",
            "Methodology",
            "Analysis & Evidence",
            "Visual Integration",
            "Conclusion",
        ]
    else:
        required = [
            "Executive Summary",
            "Analysis & Evidence",
            "Conclusion",
        ]

    headings = [h.lower() for h in extract_markdown_headings(report_text)]
    text_lower = (report_text or "").lower()

    syn = {
        "executive summary": ["tl;dr", "summary", "概要", "总结"],
        "background": ["context", "背景", "来龙去脉"],
        "methodology": ["method", "approach", "方法", "口径", "how i searched"],
        "analysis & evidence": ["analysis", "evidence", "findings", "分析", "证据", "发现"],
        "visual integration": ["visual", "figure", "chart", "image", "图", "图表", "可视化"],
        "conclusion": ["conclusion", "takeaways", "结论", "建议"],
    }

    def present(name: str) -> bool:
        if bool(section_flags.get(name)) is True:
            return True
        n = name.lower()
        if n in text_lower or any(n in h for h in headings):
            return True
        for s in syn.get(n, []):
            if s in text_lower:
                return True
        return False

    exists_flags = {k: bool(present(k)) for k in syn.keys()}
    prod = 1.0
    for k in required:
        prod *= 1.0 if exists_flags.get(k, False) else 0.0

    # Visual expectation: daily is more tolerant
    word_cnt = len(tokenize_words(report_text))
    if report_type == "research":
        target_visuals = max(1.0, word_cnt / 350.0)
    else:
        target_visuals = max(1.0, word_cnt / 700.0)

    # Count visuals (URLs in markdown image syntax)
    img_md = re.findall(r"!\[[^\]]*\]\((https?://[^)]+)\)", report_text or "", flags=re.IGNORECASE)
    img_cnt = len(set(img_md))
    W_over_Wstar = safe_div(img_cnt, target_visuals, 0.0)
    W_factor = min(1.0, W_over_Wstar)

    S = clamp01(prod * W_factor * clamp01(theta))
    stats = {
        "sections_present": exists_flags,
        "required_sections": required,
        "product_presence": prod,
        "img_cnt": img_cnt,
        "word_cnt": word_cnt,
        "target_visuals": target_visuals,
        "W_over_Wstar": W_over_Wstar,
        "W_factor": W_factor,
        "theta": theta,
    }
    return S, stats


def compute_general(report_text: str, rules_text: str, question_text: str, report_type: str) -> Dict[str, Any]:
    # deterministic features
    TTR = ttr(report_text)
    H = title_density(report_text)
    L = long_sentence_ratio(report_text)

    judge = judge_general_features(question_text, report_text, rules_text, report_type=report_type)
    E = float(judge.get("E", 0.0))

    # Readability (formula)
    a0, a1, a2, a3, a4 = 0.0, 2.0, 1.2, 2.0, 3.0
    R_formula = clamp01(logistic(a0 + a1 * TTR + a2 * H - a3 * L - a4 * E))

    # Insightfulness (formula)
    p_analysis = float(judge.get("p_analysis", 0.0))
    rho_connective = float(judge.get("rho_connective", 0.0))
    rho_synthesis = float(judge.get("rho_synthesis", 0.0))
    I_formula = clamp01((max(1e-9, p_analysis) * max(1e-9, rho_connective) * max(1e-9, rho_synthesis)) ** (1.0 / 3.0))

    # Coherence (formula)
    U = float(judge.get("U", 0.0))
    O = float(judge.get("O", 0.0))
    T = float(judge.get("T", 0.0))
    kappa = float(judge.get("kappa", 0.0))
    lambda_conflict = 2.0
    C_formula = clamp01(harmonic_mean([U, O, T]) * max(0.0, (1.0 - lambda_conflict * kappa)))

    # Structure (formula)
    theta = float(judge.get("theta", 0.0))
    section_flags = judge.get("sections", {}) if isinstance(judge.get("sections"), dict) else {}
    S_formula, S_stats = _compute_structural_completeness(report_text, section_flags, theta, report_type=report_type)

    # LLM holistic scores (0..1)
    R_llm = float(judge.get("llm_R", R_formula))
    I_llm = float(judge.get("llm_I", I_formula))
    C_llm = float(judge.get("llm_C", C_formula))
    S_llm = float(judge.get("llm_S", S_formula))
    R_llm, I_llm, C_llm, S_llm = map(clamp01, [R_llm, I_llm, C_llm, S_llm])

    alpha = _alpha_formula(report_type)
    R = alpha * R_formula + (1.0 - alpha) * R_llm
    I = alpha * I_formula + (1.0 - alpha) * I_llm
    C = alpha * C_formula + (1.0 - alpha) * C_llm
    S = alpha * S_formula + (1.0 - alpha) * S_llm

    # Dynamic weights (General)
    ctrl = judge_task_controller(question_text, report_type)
    v = ctrl.get("v_general", [0.0, 0.0, 0.0, 0.0])

    # g(doc) heuristics
    g = [0.0, 0.0, 0.0, 0.0]  # R,I,C,S
    if S < 0.5 or S_stats.get("product_presence", 1.0) == 0.0:
        g[3] += 1.0
    if E > 0.10 or L > 0.35:
        g[0] += 0.8
    if H > 0.12 and S > 0.7:
        g[1] += 0.4
        g[2] += 0.4

    # Prior pi^G: uniform (log domain)
    base_p = [0.25, 0.25, 0.25, 0.25]
    pi = [math.log(p + 1e-9) for p in base_p]

    alpha_w, beta_w = 1.0, 1.0
    logits = [pi[i] + alpha_w * float(v[i]) + beta_w * float(g[i]) for i in range(4)]
    w = softmax(logits)
    wR, wI, wC, wS = w

    GEN = 100.0 * (wR * R + wI * I + wC * C + wS * S)

    return {
        "R": float(R), "I": float(I), "C": float(C), "S": float(S),
        "score": float(GEN),
        "weights": {"w_R": float(wR), "w_I": float(wI), "w_C": float(wC), "w_S": float(wS)},
        "features": {
            "TTR": TTR, "H": H, "L": L, "E": E,
            "p_analysis": p_analysis, "rho_connective": rho_connective, "rho_synthesis": rho_synthesis,
            "U": U, "O": O, "T": T, "kappa": kappa, "theta": theta,
            "alpha_formula": alpha,
            "R_formula": R_formula, "I_formula": I_formula, "C_formula": C_formula, "S_formula": S_formula,
            "R_llm": R_llm, "I_llm": I_llm, "C_llm": C_llm, "S_llm": S_llm,
            "S_stats": S_stats,
            "ctrl": ctrl,
            "g_doc": {"R": g[0], "I": g[1], "C": g[2], "S": g[3]},
            "logits": logits,
        },
        "judge": judge,
        "ctrl": ctrl,
    }
