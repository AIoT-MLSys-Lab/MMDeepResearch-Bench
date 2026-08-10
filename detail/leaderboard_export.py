from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LEADERBOARD_COLUMNS = [
    "Model",
    "Overall",
    "Read.",
    "Insh.",
    "Stru.",
    "Vef.",
    "Con.",
    "Cov.",
    "Fid.",
    "Sem.",
    "Acc.",
    "VQA",
]

TASK_COLUMNS = [
    "Model",
    "Task ID",
    "Split",
    "Overall",
    "Stored Overall",
    "Overall Delta",
    "FLAE",
    "Read.",
    "Insh.",
    "Stru.",
    "TRACE",
    "Vef.",
    "Con.",
    "Cov.",
    "Fid.",
    "MOSAIC",
    "Sem.",
    "Acc.",
    "VQA",
    "Complete",
    "Report Path",
    "Report SHA256",
    "MOSAIC Summary Path",
    "MOSAIC Summary SHA256",
]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    valid = [v for value in values if (v := _number(value)) is not None]
    return sum(valid) / len(valid) if valid else None


def _round(value: Any, digits: int) -> Any:
    value = _number(value)
    return "" if value is None else round(value, digits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_split(qid: str, report_type: str = "") -> str:
    """Recover the released 100 Research / 40 Daily split from canonical QIDs."""
    match = re.fullmatch(r"Q(\d+)", str(qid).strip(), flags=re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 0 <= index < 100:
            return "Research"
        if 100 <= index < 140:
            return "Daily"
    return "Research" if str(report_type).lower().startswith("research") else "Daily"


def _read_records(results_jsonl: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    retained: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    duplicates: List[str] = []
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record.get("qid") or "").strip()
            if not qid:
                raise ValueError(f"Missing qid at {results_jsonl}:{line_number}")
            if qid in retained:
                duplicates.append(qid)
            else:
                order.append(qid)
            retained[qid] = record
    return [retained[qid] for qid in order], duplicates


def _scores(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("scores") or (record.get("scoring") or {}).get("scores") or {}


def _raw(record: Dict[str, Any], component: str) -> Dict[str, Any]:
    return ((record.get("scoring") or {}).get("raw") or {}).get(component) or {}


def _resolve_mm_summary(record: Dict[str, Any], run_root: Path) -> Optional[Path]:
    raw_path = (record.get("mm") or {}).get("l5_final_summary")
    if isinstance(raw_path, str) and raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate

    qid = str(record.get("qid") or "")
    candidates = [
        run_root / "reports" / "mm" / qid / "mm_router_routed" / "final_summary.json",
        run_root / "mm" / qid / "mm_router_routed" / "final_summary.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def _portable_path(path: Optional[Path], run_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _report_path(record: Dict[str, Any], run_root: Path) -> Optional[Path]:
    qid = str(record.get("qid") or "")
    canonical = run_root / "reports" / f"{qid}.md"
    if canonical.exists():
        return canonical
    raw_path = record.get("report_path") or (record.get("generation") or {}).get("report_path")
    if isinstance(raw_path, str) and raw_path and Path(raw_path).exists():
        return Path(raw_path)
    return None


def _mm_dimensions(record: Dict[str, Any], run_root: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    path = _resolve_mm_summary(record, run_root)
    if path is None:
        return {}, None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        dims = ((obj.get("summary") or {}).get("avg_metric_by_dim") or {})
        return dims if isinstance(dims, dict) else {}, path
    except (OSError, json.JSONDecodeError):
        return {}, path


def _percent(value: Any) -> Optional[float]:
    value = _number(value)
    return None if value is None else 100.0 * value


def _task_row(record: Dict[str, Any], model: str, run_root: Path) -> Dict[str, Any]:
    scores = _scores(record)
    general = _raw(record, "general")
    evidence = _raw(record, "evidence")
    dims, mm_path = _mm_dimensions(record, run_root)
    report_path = _report_path(record, run_root)

    gen = _number(scores.get("GEN"))
    evi = _number(scores.get("EVI"))
    mosaic = _number(scores.get("MM"))
    vef = _number(scores.get("VEF"))
    trace = None if evi is None or vef is None else 0.6 * evi + 0.4 * vef
    overall = None if gen is None or trace is None or mosaic is None else (
        0.2 * gen + 0.5 * trace + 0.3 * mosaic
    )
    stored_overall = _number(scores.get("FINAL_MMDR"))
    if stored_overall is None:
        stored_overall = _number(scores.get("FINAL"))

    read = _percent(general.get("R"))
    insight = _percent(general.get("I"))
    structure = _percent(general.get("S"))
    consistency = _percent(evidence.get("E_con"))
    coverage = _percent(evidence.get("E_cov"))
    fidelity = _percent(evidence.get("E_fid"))
    complete = all(value is not None for value in (
        gen, evi, mosaic, vef, read, insight, structure, consistency, coverage, fidelity
    ))

    qid = str(record.get("qid") or "")
    return {
        "Model": model,
        "Task ID": qid,
        "Split": canonical_split(qid, str(record.get("report_type") or "")),
        "Overall": overall,
        "Stored Overall": stored_overall,
        "Overall Delta": None if overall is None or stored_overall is None else overall - stored_overall,
        "FLAE": gen,
        "Read.": read,
        "Insh.": insight,
        "Stru.": structure,
        "TRACE": trace,
        # VEF is the thresholded PASS indicator stored on the 0/100 scale.
        "Vef.": vef,
        "Con.": consistency,
        "Cov.": coverage,
        "Fid.": fidelity,
        "MOSAIC": mosaic,
        "Sem.": _percent(dims.get("semantic")),
        "Acc.": _percent(dims.get("data_accuracy")),
        "VQA": _percent(dims.get("vqa_score")),
        "Complete": complete,
        "Report Path": _portable_path(report_path, run_root),
        "Report SHA256": _sha256(report_path) if report_path else "",
        "MOSAIC Summary Path": _portable_path(mm_path, run_root),
        "MOSAIC Summary SHA256": _sha256(mm_path) if mm_path else "",
    }


def _mode(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    return counts.most_common(1)[0][0] if counts else "unknown-model"


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]], digits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round(value, digits) if _number(value) is not None else value for key, value in row.items()})


def export_leaderboard_outputs(
    results_jsonl: Path,
    output_dir: Optional[Path] = None,
    model_name: str = "",
    split: str = "all",
    expected_tasks: int = 140,
    require_complete: bool = False,
) -> Dict[str, Path]:
    results_jsonl = Path(results_jsonl).resolve()
    if not results_jsonl.exists():
        raise FileNotFoundError(results_jsonl)

    records, duplicates = _read_records(results_jsonl)
    run_root = results_jsonl.parent.parent
    model = model_name.strip() or _mode(str(record.get("report_model") or "") for record in records)
    rows = [_task_row(record, model, run_root) for record in records]

    split_name = split.strip().lower()
    if split_name not in {"all", "research", "daily"}:
        raise ValueError("split must be one of: all, research, daily")
    selected = rows if split_name == "all" else [row for row in rows if row["Split"].lower() == split_name]

    complete_rows = [row for row in selected if row["Complete"]]
    expected_for_split = expected_tasks if split_name == "all" else (100 if split_name == "research" else 40)
    submission_ready = len(selected) == expected_for_split and len(complete_rows) == expected_for_split
    leaderboard = {"Model": model}
    for column in LEADERBOARD_COLUMNS[1:]:
        leaderboard[column] = _mean(row.get(column) for row in complete_rows if row.get(column) != "")

    output_dir = Path(output_dir).resolve() if output_dir else run_root / "summary"
    stem = results_jsonl.stem + ("" if split_name == "all" else f".{split_name}")
    leaderboard_suffix = "leaderboard" if submission_ready else "partial_leaderboard"
    leaderboard_path = output_dir / f"{stem}.{leaderboard_suffix}.csv"
    task_path = output_dir / f"{stem}.task_scores.csv"
    manifest_path = output_dir / f"{stem}.leaderboard_manifest.json"

    _write_csv(leaderboard_path, LEADERBOARD_COLUMNS, [leaderboard], digits=2)
    _write_csv(task_path, TASK_COLUMNS, selected, digits=6)

    counts = {
        column: sum(_number(row.get(column)) is not None for row in complete_rows)
        for column in LEADERBOARD_COLUMNS[1:]
    }
    missing_tasks = []
    required = ["Overall", "FLAE", "Read.", "Insh.", "Stru.", "TRACE", "Vef.", "Con.", "Cov.", "Fid.", "MOSAIC"]
    for row in selected:
        missing = [name for name in required if _number(row.get(name)) is None]
        if missing:
            missing_tasks.append({"task_id": row["Task ID"], "missing": missing})
    deltas = [abs(float(row["Overall Delta"])) for row in complete_rows if _number(row.get("Overall Delta")) is not None]
    manifest = {
        "schema": "mmdr-leaderboard-v1",
        "source_results": _portable_path(results_jsonl, run_root),
        "source_sha256": _sha256(results_jsonl),
        "model": model,
        "split": split_name,
        "records_selected": len(selected),
        "records_complete": len(complete_rows),
        "expected_records": expected_for_split,
        "submission_ready": submission_ready,
        "duplicate_qids_last_record_retained": duplicates,
        "missing_component_tasks": missing_tasks,
        "metric_counts": counts,
        "formula_verification": {
            "records_compared": len(deltas),
            "mean_absolute_delta": _mean(deltas),
            "max_absolute_delta": max(deltas) if deltas else None,
        },
        "formulas": {
            "TRACE": "0.6 * EVI + 0.4 * VEF",
            "Overall": "0.2 * FLAE + 0.5 * TRACE + 0.3 * MOSAIC",
            "VEF": "100 * mean(thresholded PASS indicator; threshold=6/10)",
        },
        "notes": [
            "Canonical split is Q0-Q99 Research and Q100-Q139 Daily.",
            "MOSAIC submetrics average only tasks on which that routed dimension is applicable and scored.",
            "VEF_Raw is intentionally excluded because historical caches contain both 0-10 and 0-100 raw scales.",
        ],
        "outputs": {
            "leaderboard_csv": leaderboard_path.name,
            "task_scores_csv": task_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if require_complete and not submission_ready:
        raise ValueError(
            f"Incomplete evaluation: selected={len(selected)}, complete={len(complete_rows)}, "
            f"expected={expected_for_split}. Diagnostic task/manifest outputs were written; "
            "no submission-ready leaderboard row was accepted."
        )
    return {
        "leaderboard": leaderboard_path,
        "tasks": task_path,
        "manifest": manifest_path,
    }
