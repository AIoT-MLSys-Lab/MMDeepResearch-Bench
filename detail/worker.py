import threading
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .models import Task, RuntimeConfig
from .data_loader import determine_report_type, format_task_for_llm
from .generation import generate_report_with_auto_clarification
from .scoring import score_report
# [核心] 引入判卷模块
from .accuracy import run_accuracy_check
from .utils import resolve_vertexai_urls
from .config import LLM

WRITE_LOCK = threading.Lock()

def maybe_run_mm_stage(
    *, 
    golden_path: Path, 
    work_dir: Path, 
    max_items: int, 
    dump_items_json: Path
) -> Dict[str, Any]:
    """
    尝试运行多模态修正阶段 (MM Stage)
    """
    try:
        from . import mm_stage
        fn = getattr(mm_stage, "run_mm_stage_from_golden", None)
        if not fn:
            return {"activated": True, "mm_score": 0.0, "error": "No entrypoint found in mm_stage"}
        
        # 调用 MM Stage
        return fn(
            golden_path=str(golden_path),
            work_dir=str(work_dir),
            max_items=max_items,
            dump_items_json=str(dump_items_json)
        )
    except Exception as e:
        return {"activated": True, "mm_score": 0.0, "error": f"MM Stage Error: {e}"}

def _calc_final_score(gen, evi, mm, vef, weights) -> float:
    """
    计算最终加权分
    Weights 顺序: (GEN, EVI, MM, VEF)
    """
    wg, we, wm, wv = weights
    
    # 如果没有 MM 分数 (比如纯文本任务或 MM 失败)，视作 0.0
    if mm is None:
        mm = 0.0
    
    total_w = wg + we + wm + wv
    if total_w <= 0:
        return 0.0
    
    # 加权平均
    score = (wg * float(gen) + we * float(evi) + wm * float(mm) + wv * float(vef)) / total_w
    return round(score, 3)

def process_single_task(
    idx: int,
    total_tasks: int,
    t: Task,
    run_context: Dict[str, Any],
    runtime: RuntimeConfig,
    client,
    rules_text: str,
    stop_event: threading.Event,
) -> None:
    if stop_event.is_set():
        return

    # 路径与 ID 准备
    safe_qid = (t.qid or f"q{idx+1}").replace("/", "_").replace("\\", "_")
    report_type = determine_report_type(t)
    q_for_llm = format_task_for_llm(t)

    reports_dir = run_context["reports_dir"]
    mm_dir = run_context["mm_dir"]
    fout = run_context["fout"]

    md_path = reports_dir / f"{safe_qid}.md"
    report_text = ""
    t0 = time.time()
    gen_err = None
    skip_generation = False

    # ==========================================================================
    # 1. 检查断点续传 (Resume)
    # ==========================================================================
    if md_path.exists():
        try:
            content = md_path.read_text(encoding="utf-8")
            if len(content) >= runtime.min_report_chars:
                print(f"[{idx+1}/{total_tasks}] {safe_qid}: [Resume] Found existing report.")
                report_text = content
                skip_generation = True
        except Exception:
            pass

    # ==========================================================================
    # 2. 生成报告 (Generation)
    # ==========================================================================
    if not report_text:
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: Generating...")
        try:
            report_text = generate_report_with_auto_clarification(
                client=client,
                rules_text=rules_text,
                question_text=q_for_llm,
                report_type=report_type,
                image_paths=t.images
            )
            # 写入文件
            if report_text:
                md_path.write_text(report_text, encoding="utf-8")
        except Exception as e:
            gen_err = str(e)
            print(f"[{idx+1}/{total_tasks}] {safe_qid}: ❌ Gen Error: {e}")
            # 如果生成挂了，直接跳过后续，不写入结果或者写入 0 分
            return

    # ==========================================================================
    # 2.5 解析 Vertex AI Search 重定向 URL (适配 Gemini Deep Research)
    # ==========================================================================
    if "vertexaisearch.cloud.google.com" in report_text:
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: Resolving vertexaisearch URLs...")
        report_text = resolve_vertexai_urls(report_text)
        md_path.write_text(report_text, encoding="utf-8")

    # 再次检查长度，如果太短，不进评测
    if len(report_text) < runtime.min_report_chars:
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: ⚠️ Report too short ({len(report_text)} chars). Skipping Judge.")
        # 写入一个空分记录
        with WRITE_LOCK:
            fout.write(json.dumps({
                "qid": safe_qid, 
                "error": "report_too_short", 
                "scores": {"FINAL_MMDR": 0.0}
            }, ensure_ascii=False) + "\n")
            fout.flush()
        return

    # ==========================================================================
    # 3. 规则评分 (Rule Scoring: GEN & EVI)
    # ==========================================================================
    scores_raw = {}
    score_err = None
    try:
        scores_raw = score_report(
            report_text=report_text,
            rules_text=rules_text,
            question_text=q_for_llm,
            report_type=report_type,
            golden_path=str(reports_dir),
            question_id=safe_qid
        )
    except Exception as e:
        score_err = f"Judge Crashed: {e}"
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: Rule Score Error: {e}")

    GEN = float(scores_raw.get("general_score", 0.0) or 0.0)
    EVI = float(scores_raw.get("evidence_score", 0.0) or 0.0)

    # ==========================================================================
    # 4. 多模态评分 (MM Scoring)
    # ==========================================================================
    MM = 0.0
    mm_payload: Dict[str, Any] = {"activated": False}
    # 只有当分数正常才跑 MM
    if GEN > 0 and EVI > 0:
        mm_res = maybe_run_mm_stage(
            golden_path=reports_dir / f"{safe_qid}.golden.json",
            work_dir=mm_dir / safe_qid,
            max_items=runtime.mm_max_items,
            dump_items_json=mm_dir / f"{safe_qid}.mm_items.json"
        )
        mm_payload = mm_res
        val = mm_res.get("mm_score")
        if val is not None:
            MM = float(val)

    # ==========================================================================
    # 5. VEF (Verification) - [FULLY RENAMED FROM ACC]
    # ==========================================================================
    VEF_Score = 0.0
    VEF_Raw = 0
    VEF_Verdict = "N/A"
    VEF_Reason = ""

    if t.ground_truth:
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: ⚖️ Verifying (VEF)...") 
        
        # [Strict Logic 1] Segment Calculation
        # idx < 100 为 ACADEMIC, 否则 DAILY
        segment = "DAILY" if report_type.lower().startswith("daily") else "ACADEMIC"
        
        # Call judge (renamed logic variable, implies VEF check)
        acc_res = run_accuracy_check(
            provider=runtime.judge_provider,
            model=runtime.judge_model,
            segment=segment,
            question=t.text,
            gt=t.ground_truth,
            report=report_text,
            images=t.images,
            temperature=LLM.judge_temperature
        )
        
        VEF_Raw = int(acc_res.get("score", 0) or 0)
        VEF_Verdict = acc_res.get("verdict", "FAIL").upper()
        VEF_Reason = str(acc_res.get("reason", ""))

        # [Strict Logic 3] 强制分数闸门
        # 如果原始分 < 6，强制 Fail，权重分为 0
        if VEF_Raw < 6:
            VEF_Verdict = "FAIL"
            VEF_Score = 0.0
        else:
            # 如果 >= 6，算 Pass，权重分给满分 (100.0) 以便加权
            VEF_Score = 100.0
            
        print(f"   -> Verdict: {VEF_Verdict} (Raw: {VEF_Raw}/10) -> VEF Score: {VEF_Score}")
    else:
        print(f"[{idx+1}/{total_tasks}] {safe_qid}: ⚠️ No GT. Skip VEF.")

    # ==========================================================================
    # 6. 最终结算 (Final Calculation)
    # ==========================================================================
    final_score = _calc_final_score(GEN, EVI, MM, VEF_Score, runtime.weights)

    # ==========================================================================
    # 7. 写入结果 (Output)
    # ==========================================================================
    # [KEYS UPDATED TO VEF]
    record = {
        "qid": safe_qid,
        "question_title": t.title,
        "run_id": run_context["run_id"],
        "report_type": report_type,
        "task_split": str((t.meta or {}).get("set") or ""),
        "task_difficulty": str((t.meta or {}).get("difficulty") or ""),
        "evaluator_semantics": "canonical-v2",
        "report_provider": run_context.get("report_provider", ""),
        "report_model": run_context.get("report_model", ""),
        "judge_provider": run_context.get("judge_provider", ""),
        "judge_model": run_context.get("judge_model", ""),
        "generation": {
            "time_s": round(time.time() - t0, 3),
            "error": gen_err,
            "report_path": str(md_path),
            "skipped_existing": skip_generation,
        },
        "scoring": {
            "error": score_err,
            "scores": {
                "GEN": GEN,
                "EVI": EVI,
                "MM": MM,
                "VEF": VEF_Score,    # Was ACC
                "VEF_Raw": VEF_Raw,  # Was ACC_Raw
                "FINAL_MMDR": final_score
            },
            "raw": scores_raw,
        },
        "scores": {
            "GEN": GEN,
            "EVI": EVI,
            "MM": MM,
            "VEF": VEF_Score,
            "VEF_Raw": VEF_Raw,
            "FINAL_MMDR": final_score
        },
        "mm": mm_payload,
        "accuracy_detail": {
            "verdict": VEF_Verdict,
            "reason": VEF_Reason,
            "has_gt": bool(t.ground_truth)
        },
        "report_path": str(md_path)
    }

    with WRITE_LOCK:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    print(f"[{idx+1}/{total_tasks}] {safe_qid}: ✅ DONE. Final={final_score:.2f} (VEF={VEF_Verdict})")
