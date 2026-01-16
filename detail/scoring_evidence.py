from __future__ import annotations

import math
from typing import Any, Dict, List

from .scoring_math import clamp01, harmonic_mean, safe_div, shannon_entropy, softmax
from .scoring_golden import extract_claim_pairs
from .scoring_judge import judge_evidence_pair, judge_task_controller
from .scoring_sources import classify_source_category, domain_of, source_is_authoritative, source_is_prohibited
from .scoring_web import fetch_evidence_excerpt_with_patience


def _cov_threshold(report_type: str) -> float:
    # daily: more forgiving; research: stricter
    return 0.45 if report_type != "research" else 0.55


def _soft_reliable_count(support_strength: float, tau: float) -> float:
    # soft indicator in [0,1]
    return clamp01(safe_div(support_strength, tau, 0.0))


def compute_evidence(report_text: str, rules_text: str, question_text: str, report_type: str, patience_chars: int) -> Dict[str, Any]:
    raw_pairs, pairs = extract_claim_pairs(report_text, max_pairs=40)
    if not pairs:
        # If absolutely no evidence hooks, keep near-zero but not crash
        ctrl = judge_task_controller(question_text, report_type)
        return {
            "E_con": 0.0, "E_cov": 0.0, "E_fid": 0.0, "E_div": 0.0,
            "score": 0.0,
            "weights": {"w_con": 0.40, "w_cov": 0.25, "w_fid": 0.25, "w_div": 0.10},
            "stats": {"reason": "no_claim_pairs"},
            "raw_pairs": raw_pairs,
            "audited_claims": [],
            "ctrl": ctrl,
        }

    audited: List[Dict[str, Any]] = []
    for p in pairs:
        claim = p["claim"]
        url = p["url"]

        fetched = fetch_evidence_excerpt_with_patience(url=url, claim=claim, patience_chars=patience_chars)
        excerpt = fetched["excerpt_text"]
        depth_score = float(fetched["depth_score"])
        meta = fetched["meta"]
        domain = meta.get("domain") or domain_of(url)

        # Local/direct images cannot be parsed by text judges; defer to MM and avoid penalizing text evidence too much.
        evidence_type = str(meta.get("evidence_type") or "").upper()
        skip_text_evidence = (evidence_type == "DIRECT_IMAGE_FILE") or (domain == "local")

        j = judge_evidence_pair(claim, url, excerpt, rules_text) if (not skip_text_evidence) else {
            "support_strength": 0.0,
            "hallucination_prob": 0.0,
            "inversion_prob": 0.0,
            "chain_sound_prob": 0.0,
            "source_category": "acceptable",
            "why": "DIRECT_IMAGE_FILE: skipped in text-evidence scoring (defer to MM).",
        }

        cat = j.get("source_category", "unknown")
        if cat == "unknown":
            cat = classify_source_category(domain)

        # Base support (LLM) * patience/image depth decay
        support_strength = clamp01(float(j.get("support_strength", 0.0)) * depth_score)

        # Fetch-failure soft fallback (avoid brittle zeros due to blocked pages) -- not for local/direct-image stubs
        fetch_ok = bool(meta.get("fetch_ok", False))
        if (not skip_text_evidence) and (not fetch_ok) and (not source_is_prohibited(domain)):
            support_strength = max(support_strength, 0.18)

        audited.append({
            **p,
            "domain": domain,
            "source_category": cat,
            "source_is_prohibited": bool(cat == "prohibited"),
            "source_is_authoritative": bool(cat == "authoritative"),
            "support_strength": support_strength,
            "supported_by_source": bool(support_strength >= _cov_threshold(report_type)),
            "hallucination_prob": float(j.get("hallucination_prob", 0.0)),
            "inversion_prob": float(j.get("inversion_prob", 0.0)),
            "chain_sound_prob": float(j.get("chain_sound_prob", 0.0)),
            "depth_score": depth_score,
            "evidence_meta": meta,
            "judge_why": j.get("why", ""),
            "skip_text_evidence": bool(skip_text_evidence),
        })

    # Only use non-skipped items to compute text Evidence score
    audited_text = [x for x in audited if not x.get("skip_text_evidence")]
    n = len(audited_text)

    ctrl = judge_task_controller(question_text, report_type)

    if n <= 0:
        # If everything was direct-image/local, we cannot score text evidence; keep conservative but non-crashing.
        return {
            "E_con": 0.0, "E_cov": 0.0, "E_fid": 1.0, "E_div": 0.0,
            "score": 0.0,
            "weights": {"w_con": 0.40, "w_cov": 0.25, "w_fid": 0.25, "w_div": 0.10},
            "stats": {"reason": "all_pairs_skipped_direct_image", "n_pairs_total": len(audited), "n_pairs_text": 0},
            "raw_pairs": raw_pairs,
            "audited_claims": audited,
            "ctrl": ctrl,
        }

    # E_con
    num = sum(float(x.get("relevance_weight", 1.0)) * float(x.get("support_strength", 0.0)) for x in audited_text)
    den = sum(float(x.get("relevance_weight", 1.0)) for x in audited_text)
    base_con = safe_div(num, den, 0.0)

    violation_rate = safe_div(sum(1 for x in audited_text if x.get("source_is_prohibited")), n, 0.0)
    eta = 1.0
    E_con = clamp01(base_con * (1.0 - eta * violation_rate))

    # E_cov (soft-count over key claims; exclude image-dependent claims)
    key_claims = [x for x in audited_text if x.get("is_key_claim") and (not x.get("depends_on_image"))]
    Q_key = len(key_claims)
    tau = _cov_threshold(report_type)
    if Q_key <= 0:
        E_cov = 0.0
        Q_rel_soft = 0.0
    else:
        Q_rel_soft = 0.0
        for x in key_claims:
            if x.get("source_is_prohibited"):
                continue
            Q_rel_soft += _soft_reliable_count(float(x.get("support_strength", 0.0)), tau=tau)
        E_cov = clamp01(safe_div(Q_rel_soft, float(Q_key), 0.0))

    # E_fid
    P_no_halluc = 1.0 - clamp01(sum(float(x.get("hallucination_prob", 0.0)) for x in audited_text) / max(1, n))
    P_no_inv = 1.0 - clamp01(sum(float(x.get("inversion_prob", 0.0)) for x in audited_text) / max(1, n))
    P_chain = clamp01(sum(float(x.get("chain_sound_prob", 0.0)) for x in audited_text) / max(1, n))
    E_fid = clamp01(harmonic_mean([P_no_halluc, P_no_inv, P_chain]))

    # E_div
    domain_counts: Dict[str, int] = {}
    for x in audited_text:
        d = x.get("domain") or ""
        if d:
            domain_counts[d] = domain_counts.get(d, 0) + 1
    m = len(domain_counts)
    if m <= 1:
        H_norm = 0.0
    else:
        ps = [safe_div(c, n, 0.0) for c in domain_counts.values()]
        H_norm = safe_div(shannon_entropy(ps), math.log(m), 0.0)

    m_auth = sum(1 for d in domain_counts.keys() if source_is_authoritative(d))
    m_acc = sum(1 for d in domain_counts.keys() if (not source_is_prohibited(d)))
    m_star = 3.0
    source_factor = min(1.0, safe_div(m_auth + 0.5 * max(0, m_acc - m_auth), m_star, 0.0))
    E_div = clamp01(H_norm * source_factor)

    # Dynamic weights (Evidence)
    v = ctrl.get("v_evidence", [0.0, 0.0, 0.0, 0.0])

    base_p = [0.40, 0.25, 0.25, 0.10]
    pi = [math.log(p + 1e-9) for p in base_p]

    h = [0.0, 0.0, 0.0, 0.0]  # con,cov,fid,div
    if violation_rate > 0.05:
        h[0] += 1.0
        h[3] -= 0.8
    if Q_key >= 3 and E_cov < 0.4:
        h[1] += 0.8
    if (1.0 - P_no_halluc) > 0.25 or (1.0 - P_no_inv) > 0.15:
        h[2] += 0.8

    gamma, delta = 1.0, 1.0
    logits = [pi[i] + gamma * float(v[i]) + delta * float(h[i]) for i in range(4)]
    w = softmax(logits)
    w_con, w_cov, w_fid, w_div = w

    EVI = 100.0 * (w_con * E_con + w_cov * E_cov + w_fid * E_fid + w_div * E_div)

    return {
        "E_con": float(E_con), "E_cov": float(E_cov), "E_fid": float(E_fid), "E_div": float(E_div),
        "score": float(EVI),
        "weights": {"w_con": float(w_con), "w_cov": float(w_cov), "w_fid": float(w_fid), "w_div": float(w_div)},
        "stats": {
            "n_pairs_total": len(audited),
            "n_pairs_text": n,
            "violation_rate": violation_rate,
            "Q_key_claims_text": Q_key,
            "Q_reliable_evidence_soft": Q_rel_soft,
            "cov_tau": tau,
            "P_no_hallucination": P_no_halluc,
            "P_no_inversion": P_no_inv,
            "P_chain_sound": P_chain,
            "domain_counts": domain_counts,
            "m": m,
            "m_auth": m_auth,
            "m_star": m_star,
            "h_doc": {"con": h[0], "cov": h[1], "fid": h[2], "div": h[3]},
            "logits": logits,
        },
        "raw_pairs": raw_pairs,
        "audited_claims": audited,  # keep all, including skipped direct-image
        "ctrl": ctrl,
    }
