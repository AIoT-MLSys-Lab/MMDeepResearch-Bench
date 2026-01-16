import sys
import os
import time
import threading
import json
import re
from typing import List, Tuple, Optional, Dict

def start_stop_listener(stop_event: threading.Event) -> None:
    """监听控制台输入，输入 stop/exit 停止任务"""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except Exception:
        return

    def _worker():
        print(">>> Type 'stop' + Enter to stop safely.")
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
            except Exception:
                break

            if cmd in {"stop", "exit", "quit"}:
                print("\n🛑 Stop command received. Attempting graceful shutdown...")
                stop_event.set()
                try:
                    threading.interrupt_main()
                except Exception:
                    pass
                break

    threading.Thread(target=_worker, daemon=True).start()

def strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def provider_is_gemini(p: str) -> bool:
    p = (p or "").strip().lower()
    return p in {"gemini", "google", "google_genai", "google-genai"}

def provider_is_azure(p: str) -> bool:
    p = (p or "").strip().lower()
    return p in {"azure", "azure_openai", "azure-openai"}

def provider_is_openrouter(p: str) -> bool:
    p = (p or "").strip().lower()
    return p in {"openrouter", "tongyi", "alibaba"}

def env_get_any(keys: List[str]) -> str:
    for k in keys:
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip().strip('"').strip("'")
    return ""

def _resolve_azure_profile(deployment: str) -> str:
    name = (deployment or "").strip().lower()
    mp = os.environ.get("AZURE_DEPLOYMENT_PROFILE_MAP", "").strip()
    if mp:
        try:
            rules = json.loads(mp)
            if isinstance(rules, dict):
                for pattern, profile in rules.items():
                    try:
                        if re.search(str(pattern), name):
                            return str(profile).strip()
                    except Exception:
                        continue
        except Exception:
            pass
    if name.startswith("claude-"):
        return "AZURE_CLAUDE"
    return "AZURE_OPENAI"

def resolve_azure_config_for_deployment(deployment: str) -> Dict[str, str]:
    profile = _resolve_azure_profile(deployment)
    api_key = env_get_any([f"{profile}_API_KEY", "AZURE_OPENAI_API_KEY"])
    endpoint = env_get_any([
        f"{profile}_ENDPOINT", f"{profile}_AZURE_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_AZURE_ENDPOINT", "AZURE_OPENAI_BASE_URL",
    ])
    api_version = env_get_any([f"{profile}_API_VERSION", "AZURE_OPENAI_API_VERSION"]) or "2024-02-15-preview"
    return {"profile": profile, "api_key": api_key, "endpoint": endpoint, "api_version": api_version}

def sanitize_run_id(run_id: str) -> str:
    run_id = (run_id or "").strip()
    if not run_id:
        return ""
    run_id = re.sub(r"[^A-Za-z0-9._\-]+", "_", run_id)
    run_id = run_id.strip("._-")
    return run_id[:80] if run_id else ""

def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"

def parse_weights(s: str) -> Tuple[float, float, float, float]:
    """
    解析权重字符串。
    支持格式: "GEN,EVI,MM,ACC" (4个值)
    如果只给了3个值，ACC 默认为 0.0
    默认值: (2.0, 3.0, 3.0, 2.0)
    """
    s = (s or "").strip()
    if not s:
        return (2.0, 3.0, 3.0, 2.0)
    
    parts = []
    try:
        parts = [float(p.strip()) for p in s.split(",") if p.strip()]
    except ValueError:
        pass

    if len(parts) == 4:
        return (parts[0], parts[1], parts[2], parts[3])
    
    # 兼容旧版本只传3个参数的情况
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2], 0.0)
        
    return (2.0, 3.0, 3.0, 2.0)

def final_weighted(gen: float, evi: float, mm: Optional[float], acc: float, weights: Tuple[float, float, float, float]) -> Tuple[float, str]:
    """
    计算最终加权分。
    weights: (w_gen, w_evi, w_mm, w_acc)
    """
    w_gen, w_evi, w_mm, w_acc = weights
    
    val_mm = float(mm) if mm is not None else 0.0
    
    denom = w_gen + w_evi + w_mm + w_acc
    if denom <= 0:
        return (0.0, "error")
        
    val = (w_gen * gen + w_evi * evi + w_mm * val_mm + w_acc * acc) / denom
    return (val, "full_weighted")

def get_gemini_api_key_from_env_or_prompt() -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return strip_quotes(api_key)
    if not os.isatty(0):
        return None
    key = input("Enter your Gemini API Key: ")
    return strip_quotes(key) or None