import argparse
import os

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # ==========================================================================
    # Run Control
    # ==========================================================================
    ap.add_argument("--run_id", type=str, default="", help="Unique run id.")
    ap.add_argument("--run_root", type=str, default="", help="Output root dir.")
    ap.add_argument("--append", action="store_true", help="Append mode.")
    ap.add_argument("--hard_exit", action="store_true", help="Force exit.")

    # ==========================================================================
    # Debug / Subset
    # ==========================================================================
    ap.add_argument("--quiz_first", action="store_true", help="Run only 1st task.")
    ap.add_argument("--quiz_index", type=int, default=0, help="Run specific task index.")

    # ==========================================================================
    # Dataset (自动从 ENV 读取)
    # ==========================================================================
    # 题目文件
    ap.add_argument("--research_json", type=str, 
                    default=os.environ.get("MMDR_RESEARCH_JSON", r"quiz.jsonl"))
    
    # 图片根目录
    ap.add_argument("--research_image_root", type=str, 
                    default=os.environ.get("MMDR_RESEARCH_IMAGE_ROOT", r"images"))
    
    # [新增] GT 文件 (答案)
    ap.add_argument("--gt_path", type=str, 
                    default=os.environ.get("MMDR_GT_FILE", r"quiz_vef.jsonl"),
                    help="Path to Ground Truth jsonl")

    # ==========================================================================
    # Rules & Config
    # ==========================================================================
    ap.add_argument("--rules_path", type=str, default="")

    # ==========================================================================
    # LLM Config (从 ENV 读取)
    # ==========================================================================
    # 1. 生成报告的模型
    ap.add_argument("--report_provider", type=str, 
                    default=os.environ.get("MMDR_REPORT_PROVIDER", "gemini"))
    ap.add_argument("--report_model", type=str, 
                    default=os.environ.get("MMDR_REPORT_MODEL", "gemini-2.5-pro"))
    
    # 2. 判卷模型 (Accuracy Judge)
    ap.add_argument("--judge_provider", type=str, 
                    default=os.environ.get("JUDGE_PROVIDER", "azure"))
    ap.add_argument("--judge_model", type=str, 
                    default=os.environ.get("JUDGE_MODEL", "gpt-4o"))

    # ==========================================================================
    # Tuning & Thresholds
    # ==========================================================================
    ap.add_argument("--max_workers", type=int, default=1)
    ap.add_argument("--min_report_chars", type=int, default=1000)
    
    # Retry Logic
    ap.add_argument("--gen_max_retries", type=int, default=3)
    ap.add_argument("--gen_retry_wait_s", type=float, default=10.0)
    ap.add_argument("--gen_between_attempts_s", type=float, default=2.0)
    
    # MM Stage
    ap.add_argument("--mm_max_items", type=int, default=12)

    # Weights: (GEN, EVI, MM, ACC) - 从 ENV 读取
    ap.add_argument("--weights", type=str, 
                    default=os.environ.get("MMDR_WEIGHTS", "2,3,3,2"), 
                    help="Weights: 'GEN,EVI,MM,ACC'")

    ap.add_argument("--no_reverse", action="store_true")

    return ap.parse_args()