from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, DefaultDict
from collections import defaultdict

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x))

def _avg(xs: List[float]) -> float | None:
    valid_xs = [float(x) for x in xs if _is_number(x)]
    return (sum(valid_xs) / len(valid_xs)) if valid_xs else None

def _fmt(x: float | None, digits: int = 2) -> str:
    if x is None: return "N/A"
    return f"{x:.{digits}f}"

def _flatten_numbers(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if obj is None: return out
    if _is_number(obj):
        out[prefix or "value"] = float(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = str(k)
            p = f"{prefix}.{kk}" if prefix else kk
            out.update(_flatten_numbers(v, p))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_flatten_numbers(v, p))
        return out
    return out

def _group_name(report_type: str) -> str:
    rt = (report_type or "").lower().strip()
    if rt.startswith("research"): return "research"
    if rt.startswith("daily"): return "daily"
    return "daily"

def _extract_general_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    g = raw.get("general")
    if not isinstance(g, dict): return {}
    out: Dict[str, float] = {}
    keep = {"R": g.get("R"), "I": g.get("I"), "C": g.get("C"), "S": g.get("S"), "score": g.get("score")}
    out.update(_flatten_numbers(keep, prefix="gen_rule"))
    return out

def _extract_evidence_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    e = raw.get("evidence")
    if not isinstance(e, dict): return {}
    out: Dict[str, float] = {}
    keep = {"E_con": e.get("E_con"), "E_cov": e.get("E_cov"), "E_fid": e.get("E_fid"), "score": e.get("score")}
    out.update(_flatten_numbers(keep, prefix="evi_rule"))
    return out

def _extract_mm_metrics_from_final_summary(path: str) -> Dict[str, float]:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        summary = obj.get("summary")
        if isinstance(summary, dict): return _flatten_numbers(summary, prefix="mm")
    except: pass
    return {}

def summarize_results(results_jsonl: Path) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {"daily": [], "research": [], "all": []}

    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    g = _group_name(rec.get("report_type", "daily"))
                    groups[g].append(rec)
                    groups["all"].append(rec)
                except Exception: continue

    def compute_group(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        gen_list, evi_list, mm_list, final_list = [], [], [], []
        sub_general: DefaultDict[str, List[float]] = defaultdict(list)
        sub_evidence: DefaultDict[str, List[float]] = defaultdict(list)
        sub_mm: DefaultDict[str, List[float]] = defaultdict(list)
        pass_count = 0
        total_judged = 0

        for r in recs:
            scores = r.get("scores")
            if not scores: scores = (r.get("scoring") or {}).get("scores") or {}

            GEN = scores.get("GEN")
            OLD_EVI = scores.get("EVI") 
            MM = scores.get("MM")
            
            # [KEY CHANGE] Look for VEF, fallback to ACC
            VEF = scores.get("VEF") or scores.get("ACC")
            VEF_RAW = scores.get("VEF_Raw") or scores.get("ACC_Raw")
            
            FINAL = scores.get("FINAL_MMDR") or scores.get("FINAL")

            if _is_number(GEN): gen_list.append(float(GEN))
            if _is_number(MM): mm_list.append(float(MM))
            if _is_number(FINAL): final_list.append(float(FINAL))

            # --- Composite Evidence Logic (Rule * 3 + VEF * 2) / 5 ---
            val_evi = float(OLD_EVI) if _is_number(OLD_EVI) else 0.0
            combined_evi = val_evi
            
            if _is_number(VEF):
                val_vef = float(VEF)
                combined_evi = (val_evi * 3.0 + val_vef * 2.0) / 5.0
                sub_evidence["VEF_Weighted"].append(val_vef)
            
            if _is_number(VEF_RAW):
                sub_evidence["VEF_Raw_Judge"].append(float(VEF_RAW))

            if _is_number(OLD_EVI) or _is_number(VEF):
                evi_list.append(combined_evi)
                sub_evidence["Rule_Score (Old EVI)"].append(val_evi)

            acc_detail = r.get("accuracy_detail") or {}
            verdict = acc_detail.get("verdict", "N/A")
            if verdict != "N/A":
                total_judged += 1
                if verdict == "PASS": pass_count += 1

            raw = (r.get("scoring") or {}).get("raw")
            if isinstance(raw, dict):
                g_metrics = _extract_general_metrics(raw)
                for k, v in g_metrics.items(): sub_general[k].append(v)
                e_metrics = _extract_evidence_metrics(raw)
                for k, v in e_metrics.items(): sub_evidence[k].append(v)

            mm_data = r.get("mm") or {}
            l5_path = mm_data.get("l5_final_summary")
            if isinstance(l5_path, str):
                mm_metrics = _extract_mm_metrics_from_final_summary(l5_path)
                for k, v in mm_metrics.items(): sub_mm[k].append(v)

        pass_rate = (pass_count / total_judged * 100.0) if total_judged > 0 else 0.0

        return {
            "n": len(recs),
            "avg": {
                "GEN": _avg(gen_list),
                "EVI": _avg(evi_list),
                "MM": _avg(mm_list),
                "FINAL_MMDR": _avg(final_list),
            },
            "pass_stats": {"rate": pass_rate, "passed": pass_count, "total": total_judged},
            "submetrics": {
                "general": {k: _avg(vs) for k, vs in sub_general.items() if _avg(vs) is not None},
                "evidence": {k: _avg(vs) for k, vs in sub_evidence.items() if _avg(vs) is not None},
                "mm": {k: _avg(vs) for k, vs in sub_mm.items() if _avg(vs) is not None},
            },
        }

    return {
        "daily": compute_group(groups["daily"]),
        "research": compute_group(groups["research"]),
        "all": compute_group(groups["all"]),
    }

def write_summary_txt(results_jsonl: Path, out_txt: Path, meta: Dict[str, Any] | None = None) -> None:
    meta = meta or {}
    agg = summarize_results(results_jsonl)

    def dump_metrics(title: str, d: Dict[str, float | None]) -> List[str]:
        lines = [f"\n{title}:"]
        if not d: lines.append("  (None)")
        for k in sorted(d.keys()):
            v = d[k]
            if v is not None: lines.append(f"  - {k}: {_fmt(float(v), 2)}")
        return lines

    def write_block(name: str, obj: Dict[str, Any]) -> List[str]:
        lines = []
        n = obj['n']
        pr = obj['pass_stats']
        
        lines.append(f"\n[{name}] n={n}")
        lines.append(f"AVG GEN:   {_fmt(obj['avg']['GEN'])}")
        # [DISPLAY CHANGE] Explicitly mention VEF
        lines.append(f"AVG EVI:   {_fmt(obj['avg']['EVI'])} (Composite: 60% Rule + 40% VEF)")
        lines.append(f"AVG MM:    {_fmt(obj['avg']['MM'])}")
        lines.append(f"AVG FINAL: {_fmt(obj['avg']['FINAL_MMDR'])}")
        
        if pr['total'] > 0:
            lines.append(f"PASS RATE: {pr['rate']:.1f}% ({pr['passed']}/{pr['total']}) [VEF]")

        lines.extend(dump_metrics("General Details", obj["submetrics"]["general"]))
        lines.extend(dump_metrics("Evidence Details (Rule + VEF)", obj["submetrics"]["evidence"]))
        lines.extend(dump_metrics("MM Details", obj["submetrics"]["mm"]))
        lines.append("-" * 60)
        return lines

    lines = ["MM-DeepResearch Evaluation Summary", "=" * 60]
    if meta:
        for k, v in meta.items(): lines.append(f"{k}: {v}")
    lines.append("-" * 60)

    lines.extend(write_block("ALL", agg["all"]))
    lines.extend(write_block("DAILY", agg["daily"]))
    lines.extend(write_block("RESEARCH", agg["research"]))

    lines.append("\nNote: 'AVG EVI' is a composite score of Rule-based Evidence (weight 3) and VEF/Accuracy (weight 2).")
    
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    
    out_json = out_txt.with_suffix(".json")
    out_json.write_text(json.dumps({"meta": meta, "aggregates": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to: {out_txt}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        write_summary_txt(Path(sys.argv[1]), Path(sys.argv[2]))