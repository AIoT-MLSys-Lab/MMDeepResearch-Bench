from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class Task:
    qid: str
    title: str
    text: str
    images: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    # [新增] 存放 Ground Truth 文本
    ground_truth: str = ""

@dataclass
class RuntimeConfig:
    max_workers: int
    min_report_chars: int
    gen_max_retries: int
    gen_retry_wait_s: float
    gen_between_attempts_s: float
    mm_max_items: int
    # 这里的 weights 含义变为 (GEN, EVI, MM, ACC) 四项
    weights: Tuple[float, float, float, float] 
    reverse_tasks: bool
    hard_exit: bool
    # [新增] 判卷相关配置
    judge_provider: str
    judge_model: str