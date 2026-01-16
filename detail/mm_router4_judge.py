# mm_router4_judge_tiny_v3.py
# Layer-4: 将底层引擎的详细结果压缩为“每张图一个分数”的简报
# V3 Update: 适配双轨制评估体系，并修复 Diagram 评分可能存在的双重计算问题

from __future__ import annotations
import json, os, re, argparse
from typing import Any, Dict, List
from collections import defaultdict

def _clamp(x: Any) -> float:
    try:
        val = float(x) if x is not None else 0.0
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0

def _score5(x01: float) -> int:
    """0~1 -> 0~5（四舍五入）"""
    return int(round(_clamp(x01) * 5))

def _short(s: Any, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def calculate_total_score(item: Dict[str, Any]) -> float:
    """
    核心修改：根据 mode 选择不同的评分公式。
    对于 Diagram，直接使用原子指标 (VQA, Hallu) 重组分数，防止语义分中包含幻觉导致重复扣分。
    """
    engine_out = item.get("engine_output") or {}
    metrics = engine_out.get("metrics") or {}
    mode = engine_out.get("mode", "visual_eval")
    
    # 获取通用指标
    fmt = _clamp(metrics.get("formatting", 1.0))

    # === Track A: 数据评估轨道 (Chart / Table) ===
    # 逻辑：Data Accuracy (90%) + Formatting (10%)
    if mode == "data_eval":
        data_acc = _clamp(metrics.get("data_accuracy"))
        
        # 熔断机制：数据完全错误则总分归零
        if data_acc < 1e-9:
            return 0.0
        
        raw_score = (data_acc * 0.9) + (fmt * 0.1)
        return raw_score * 5.0

    # === Track B: 视觉评估轨道 (Diagram / Photo) ===
    else:
        # 判断是否为 Diagram 且具备详细指标
        img_type = item.get("type", "photo").lower()
        hallu_data = metrics.get("diagram_hallu") or {}
        vqa_val = metrics.get("vqa_score")
        
        # --- 修正逻辑：Diagram 显式重算 ---
        # 如果是 Diagram 且有幻觉分，忽略 metrics['semantic']，防止双重计算
        if img_type == "diagram" and vqa_val is not None and "score" in hallu_data:
            # 1. Semantic (语义) -> 由 VQA Score 接管 (权重 0.5)
            # 2. Faithful (忠实) -> 由 Hallu Score 接管 (权重 0.4)
            # 3. Formatting (格式) -> 权重 0.1
            
            vqa_s = _clamp(vqa_val)
            hallu_s = _clamp(hallu_data["score"]) # score越高代表幻觉越少(1-rate)
            
            # 熔断：如果 VQA 和 Hallu 都很差
            if vqa_s < 1e-9 and hallu_s < 1e-9:
                return 0.0

            raw_score = (vqa_s * 0.5) + (hallu_s * 0.4) + (fmt * 0.1)
            return raw_score * 5.0

        # --- 默认逻辑：Photo 或 简单模式 ---
        # 直接使用 Engine 计算好的 Semantic/Faithful
        else:
            sem = _clamp(metrics.get("semantic"))
            fai = _clamp(metrics.get("faithful"))
            
            if sem < 1e-9 and fai < 1e-9:
                return 0.0
            
            # Photo: Semantic=VQA, Faithful=VQA. 相当于 0.9*VQA + 0.1*Fmt
            raw_score = (sem * 0.5) + (fai * 0.4) + (fmt * 0.1)
            return raw_score * 5.0

def compact_layer4(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    压缩数据，生成精简报告
    """
    out: List[Dict[str, Any]] = []
    
    for it in items:
        # 1. 计算总分
        raw_total = calculate_total_score(it)
        total_int = int(round(raw_total))
        
        engine_out = it.get("engine_output") or {}
        metrics = engine_out.get("metrics") or {}
        mode = engine_out.get("mode", "visual_eval")
        
        # 2. 提取关键细节 (Details)
        details_str = ""
        if total_int <= 1:
            details_str = "Quality too low."
        elif mode == "data_eval":
            det = metrics.get("details") or {}
            n_sup = det.get("n_supported", 0)
            n_tot = det.get("n_total", 0)
            acc = metrics.get("data_accuracy", 0.0)
            details_str = f"Claims: {n_sup}/{n_tot} supported ({acc:.0%})"
        else:
            hallu = metrics.get("diagram_hallu") or {}
            if hallu and "hallucination_rate" in hallu:
                hr = hallu["hallucination_rate"]
                if hr is not None:
                    details_str = f"Hallu Rate: {hr:.1%}"
                elif "error" in hallu:
                    details_str = f"Hallu Error: {hallu['error']}"
            else:
                sem = metrics.get("semantic", 0.0)
                details_str = f"Semantic Score: {sem:.2f}"

        # 3. 构造输出
        # 为了展示方便，我们把用于计算的主要指标取出来
        # Diagram: Main=VQA, Sec=Hallu
        # Photo: Main=Semantic, Sec=Faithful
        # Data: Main=Accuracy, Sec=Formatting
        if mode == "data_eval":
            score_main = metrics.get("data_accuracy")
            score_sec = metrics.get("formatting")
        elif it.get("type") == "diagram" and metrics.get("vqa_score") is not None:
            score_main = metrics.get("vqa_score")
            score_sec = (metrics.get("diagram_hallu") or {}).get("score")
        else:
            score_main = metrics.get("semantic")
            score_sec = metrics.get("faithful")

        rec = {
            "index": it.get("index"),
            "article_id": it.get("source_file", "unknown"),
            "page": it.get("page"),
            "type": it.get("type", "photo"),
            "image_path": it.get("image_path", ""),
            "caption_start": _short(it.get("caption", ""), 30),
            "total": total_int,
            "mode": mode,
            "score_main": _score5(score_main),
            "score_sec":  _score5(score_sec),
            "details": details_str
        }
        out.append(rec)
        
    return out

def calculate_article_stats(processed_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for item in processed_items:
        aid = item.get("article_id", "unknown")
        grouped[aid].append(item)
    
    summary = {}
    for aid, img_list in grouped.items():
        if not img_list: continue
        scores = [x["total"] for x in img_list]
        avg_score = sum(scores) / len(scores)
        dist = {i: 0 for i in range(6)}
        for s in scores: dist[s] += 1
        bad_images = [x["index"] for x in img_list if x["total"] <= 1]
        
        summary[aid] = {
            "n_images": len(img_list),
            "avg_score": float(f"{avg_score:.2f}"),
            "score_distribution": dist,
            "low_score_ratio": float(f"{len(bad_images) / len(img_list):.2f}"),
            "issues_indices": bad_images
        }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default="mm_router_routed/all_with_results.json")
    ap.add_argument("--out_json", default="mm_router_routed/mm_layer4_scores_tiny.json")
    ap.add_argument("--only_summary", action="store_true", help="只输出文章汇总信息")
    args = ap.parse_args()

    if not os.path.exists(args.in_json):
        print(f"❌ 错误: 找不到输入文件: {args.in_json}")
        return

    print(f"📖 读取文件: {args.in_json} ...")
    with open(args.in_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    processed_images = compact_layer4(items)
    article_summary = calculate_article_stats(processed_images)

    final_output = {"summary": article_summary, "images": processed_images}
    output_data = article_summary if args.only_summary else final_output

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成。共 {len(processed_images)} 张图片，来自 {len(article_summary)} 篇文章。")
    print("-" * 80)
    print(f"{'Article ID':<30} | {'Count':<5} | {'Avg Score':<10} | {'Pass Rate':<10}")
    print("-" * 80)
    for aid, stats in sorted(article_summary.items()):
        pass_rate = (1.0 - stats['low_score_ratio']) * 100
        print(f"{_short(aid, 30):<30} | {stats['n_images']:<5} | {stats['avg_score']:<10.2f} | {pass_rate:5.1f}%")
    print("-" * 80)
    print(f"📁 结果已写入: {args.out_json}")

if __name__ == "__main__":
    main()