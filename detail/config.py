# config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _strip_quotes(v: str) -> str:
    v = (v or "").strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v


def _env_str(name: str) -> str | None:
    v = os.environ.get(name)
    if not v:
        return None
    v = _strip_quotes(v)
    return v or None


def _env_int(name: str) -> int | None:
    v = _env_str(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _env_float(name: str) -> float | None:
    v = _env_str(name)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _env_path(name: str, default: Path) -> Path:
    v = _env_str(name)
    if not v:
        return default
    try:
        return Path(v).expanduser().resolve()
    except Exception:
        return default


def _default_root() -> Path:
    env = os.environ.get("MMDR_ROOT")
    if env:
        try:
            return Path(_strip_quotes(env)).expanduser().resolve()
        except Exception:
            pass
    return Path(__file__).resolve().parent


ROOT_DIR = _default_root()


@dataclass(frozen=True)
class PathConfig:
    # ---------- core ----------
    rules_path: Path = _env_path("MMDR_RULES_PATH", ROOT_DIR / "Rules.txt")
    question_docx_path: Path = _env_path("MMDR_QUESTION_DOCX_PATH", ROOT_DIR / "question set.docx")

    # ---------- research set ----------
    # (与你 run.py 里用的 env 名保持一致：MMDR_RESEARCH_JSON / MMDR_RESEARCH_IMAGE_ROOT)
    research_json_path: Path = _env_path(
        "MMDR_RESEARCH_JSON",
        ROOT_DIR / "research questions" / "quiz" / "quiz.json",
    )
    research_image_root: Path = _env_path(
        "MMDR_RESEARCH_IMAGE_ROOT",
        ROOT_DIR / "research questions" / "quiz",
    )

    # ---------- outputs ----------
    # 新版推荐用 runs_root 组织多次运行：runs_root/<run_id>/{reports,results,summary}
    runs_root: Path = _env_path("MMDR_RUNS_ROOT", ROOT_DIR / "reports_runs")

    # 这两个保留给旧代码/脚本使用（不影响你现在的 run_id 结构）
    reports_dir: Path = _env_path("MMDR_REPORTS_DIR", ROOT_DIR / "reports")
    results_path: Path = _env_path("MMDR_RESULTS_PATH", ROOT_DIR / "results.jsonl")


@dataclass(frozen=True)
class CrawlConfig:
    """Used by scoring_web.py / scoring.py for fetching cited pages."""
    user_agent: str = (
        _env_str("MMDR_CRAWL_UA")
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/129.0.0.0 Safari/537.36"
    )
    timeout_s: int = _env_int("MMDR_CRAWL_TIMEOUT_S") or 15
    max_bytes: int = _env_int("MMDR_CRAWL_MAX_BYTES") or 2_000_000

    patience_chars_daily: int = _env_int("MMDR_PATIENCE_DAILY") or 90_000
    patience_chars_research: int = _env_int("MMDR_PATIENCE_RESEARCH") or 140_000

    ideal_img_rank: int = _env_int("MMDR_IDEAL_IMG_RANK") or 3
    max_img_rank: int = _env_int("MMDR_MAX_IMG_RANK") or 25


@dataclass(frozen=True)
class LLMConfig:
    # Report LLM（默认值 + ENV 覆盖；CLI 也能在主脚本里 object.__setattr__ 覆盖）
    report_provider: str = (_env_str("MMDR_REPORT_PROVIDER") or "gemini")
    report_model: str = (_env_str("MMDR_REPORT_MODEL") or "gemini-2.5-pro")

    # Judge LLM（默认 Gemini 2.5 Pro；需要 Azure/gpt-5.2 时走 CLI/ENV）
    judge_provider: str = (_env_str("MMDR_JUDGE_PROVIDER") or "gemini")
    judge_model: str = (_env_str("MMDR_JUDGE_MODEL") or "gemini-2.5-pro")

    temperature: float = _env_float("MMDR_REPORT_TEMPERATURE") or 0.7
    max_output_tokens: int = _env_int("MMDR_REPORT_MAX_TOKENS") or 8192

    judge_temperature: float = _env_float("MMDR_JUDGE_TEMPERATURE") or 0.1
    max_judge_tokens: int = _env_int("MMDR_JUDGE_MAX_TOKENS") or 4096


# IMPORTANT: 这些名字要导出，旧代码才不会炸
PATHS = PathConfig()
CRAWL = CrawlConfig()
LLM = LLMConfig()
