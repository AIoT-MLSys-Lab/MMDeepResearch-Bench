import os
import time
import json
import traceback
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

# Relative imports within the detail package
from . import cli, utils, data_loader, worker
from .models import Task, RuntimeConfig
from .config import PATHS, LLM
from .generation import load_rules_text
from .results_summary import write_summary_txt
from .llm_interface import GeminiChatClient, AzureOpenAIChatClient, OpenRouterChatClient, DummyEchoClient

def _maybe_set_llm_attr(name: str, value: Optional[str]) -> None:
    if value is None:
        return
    v = str(value).strip()
    if not v:
        return
    object.__setattr__(LLM, name, v)

def run_pipeline_logic() -> None:
    args = cli.parse_args()

    # Apply optional CLI LLM config
    _maybe_set_llm_attr("report_provider", args.report_provider or None)
    _maybe_set_llm_attr("report_model", args.report_model or None)
    _maybe_set_llm_attr("judge_provider", args.judge_provider or None)
    _maybe_set_llm_attr("judge_model", args.judge_model or None)

    run_id = utils.sanitize_run_id(args.run_id) or utils.sanitize_run_id(os.environ.get("MMDR_RUN_ID", "")) or utils._default_run_id()

    # Path Handling: Go up 2 levels from detail/controller.py to get project root
    project_root = Path(__file__).resolve().parent.parent
    base_root = Path(args.run_root).expanduser().resolve() if args.run_root else (project_root / "reports_runs")
    
    run_root = base_root / run_id
    reports_dir = run_root / "reports"
    results_dir = run_root / "results"
    summary_dir = run_root / "summary"
    mm_dir = reports_dir / "mm"

    for d in [reports_dir, results_dir, summary_dir, mm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{run_id}.jsonl"
    summary_path = summary_dir / f"{run_id}.txt"

    # [Config Update] weights now supports 4 items (GEN, EVI, MM, VEF)
    runtime = RuntimeConfig(
        max_workers=max(1, int(args.max_workers)),
        min_report_chars=max(1, int(args.min_report_chars)),
        gen_max_retries=max(1, int(args.gen_max_retries)),
        gen_retry_wait_s=max(0.0, float(args.gen_retry_wait_s)),
        gen_between_attempts_s=max(0.0, float(args.gen_between_attempts_s)),
        mm_max_items=max(1, int(args.mm_max_items)),
        weights=utils.parse_weights(args.weights),
        reverse_tasks=(not bool(args.no_reverse)),
        hard_exit=bool(args.hard_exit),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model
    )

    # Gemini key setup (if needed)
    if utils.provider_is_gemini(LLM.report_provider) or utils.provider_is_gemini(LLM.judge_provider):
        api_key = utils.get_gemini_api_key_from_env_or_prompt()
        if api_key:
            from .api import set_gemini_api_key
            set_gemini_api_key(api_key)

    # Init Client
    try:
        if utils.provider_is_gemini(LLM.report_provider):
            print(f"Initializing Gemini Client (report_model={LLM.report_model})...")
            client = GeminiChatClient(model_name=LLM.report_model)
        elif utils.provider_is_azure(LLM.report_provider):
            cfg = utils.resolve_azure_config_for_deployment(LLM.report_model)
            print(f"Initializing Azure Client (deployment={LLM.report_model}) via {cfg['profile']} -> {cfg['endpoint']}")
            client = AzureOpenAIChatClient(
                deployment_name=LLM.report_model,
                api_key=cfg["api_key"],
                azure_endpoint=cfg["endpoint"],
                api_version=cfg["api_version"],
            )
        elif utils.provider_is_openrouter(LLM.report_provider):
            print(f"Initializing OpenRouter Client (report_model={LLM.report_model})...")
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TONGYI_API_KEY") or ""
            client = OpenRouterChatClient(model_name=LLM.report_model, api_key=api_key)
        else:
            print("Using Dummy Client (no valid provider configured)")
            client = DummyEchoClient()
    except Exception as e:
        print(f"❌ Init Client Failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print("   Falling back to DummyEchoClient")
        client = DummyEchoClient()

    if hasattr(client, "vision_enabled"):
        print(f">>> Client vision support: {getattr(client, 'vision_enabled', None)}")

    # Load Rules
    rules_path = Path(args.rules_path).expanduser().resolve() if args.rules_path else Path(PATHS.rules_path)
    rules_text = load_rules_text(rules_path)

    # Load Tasks
    tasks: List[Task] = []
    research_json = args.research_json
    research_image_root = args.research_image_root
    gt_path = args.gt_path  # [Important] Pass GT path

    if getattr(args, "quiz_first", False) or (getattr(args, "quiz_index", 0) > 0):
        research_tasks = data_loader.load_research_tasks(
            research_json, 
            research_image_root, 
            gt_path=gt_path
        )
        if not research_tasks:
            raise RuntimeError(f"No research tasks loaded from: {research_json}")

        pick = 0 if getattr(args, "quiz_first", False) else int(getattr(args, "quiz_index", 0)) - 1
        if pick < 0 or pick >= len(research_tasks):
            raise RuntimeError(f"--quiz_index out of range: {pick+1}, total={len(research_tasks)}")

        tasks = [research_tasks[pick]]
        print(f">>> [Quiz Debug] Running ONLY quiz.json item #{pick+1}: qid={tasks[0].qid}")
    else:
        if research_json:
            tasks.extend(data_loader.load_research_tasks(
                research_json, 
                research_image_root,
                gt_path=gt_path
            ))

    if runtime.reverse_tasks:
        tasks.reverse()

    print(f"RUN_ID: {run_id}")
    print(f"Run root: {run_root}")
    print(f"Concurrency: {runtime.max_workers} workers")
    print(f"Report provider/model: {LLM.report_provider} / {LLM.report_model}")
    print(f"Judge provider/model:  {LLM.judge_provider} / {LLM.judge_model}")
    print(f"Min report chars gate: {runtime.min_report_chars}")
    # [FIXED HERE] Updated display name from ACC to VEF
    print(f"Weights (GEN,EVI,MM,VEF): {runtime.weights}")

    # Check for existing finished tasks
    finished_qids = set()
    if args.append and results_path.exists():
        print("Scanning existing results...")
        try:
            with results_path.open("r", encoding="utf-8") as f_read:
                for line in f_read:
                    if line.strip():
                        try:
                            finished_qids.add(json.loads(line)["qid"])
                        except Exception:
                            pass
        except Exception:
            pass
    print(f" -> Found {len(finished_qids)} finished tasks. Skipping them.")

    run_context = {
        "run_id": run_id,
        "run_root": run_root,
        "reports_dir": reports_dir,
        "mm_dir": mm_dir,
        "report_provider": LLM.report_provider,
        "report_model": LLM.report_model,
        "judge_provider": LLM.judge_provider,
        "judge_model": LLM.judge_model,
        "fout": None,
    }

    stop_event = threading.Event()
    utils.start_stop_listener(stop_event)

    futures = []
    status = "Finished"

    file_mode = "a" if args.append else "w"
    try:
        with results_path.open(file_mode, encoding="utf-8") as fout:
            run_context["fout"] = fout
            executor = ThreadPoolExecutor(max_workers=runtime.max_workers)
            try:
                for idx, t in enumerate(tasks):
                    safe_qid = (t.qid or f"q{idx+1}").replace("/", "_").replace("\\", "_")
                    if safe_qid in finished_qids:
                        continue

                    futures.append(
                        executor.submit(
                            worker.process_single_task,
                            idx,
                            len(tasks),
                            t,
                            run_context,
                            runtime,
                            client,
                            rules_text,
                            stop_event,
                        )
                    )

                print(f" -> Submitted {len(futures)} tasks.")
                print(" -> Press Ctrl+C OR type 'stop' + Enter to stop safely.")

                while True:
                    if stop_event.is_set():
                        raise KeyboardInterrupt
                    if all(f.done() for f in futures):
                        break
                    time.sleep(0.5)

            except KeyboardInterrupt:
                status = "Interrupted"
                print("\n\n🛑 Stop requested. Signaling workers to stop...")
                stop_event.set()
                for f in futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    executor.shutdown(wait=False)

                t_end = time.time() + 5.0
                while time.time() < t_end and any(not f.done() for f in futures):
                    time.sleep(0.2)
                print(" -> Exiting main loop now.")

            finally:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    try:
                        executor.shutdown(wait=False)
                    except Exception:
                        pass

    except Exception as e:
        status = "Crashed"
        print(f"Runtime error: {e}")
        traceback.print_exc()

    write_summary_txt(results_jsonl=results_path, out_txt=summary_path, meta={"run_id": run_id, "status": status})
    print("\nDone.")

    if runtime.hard_exit:
        os._exit(0)