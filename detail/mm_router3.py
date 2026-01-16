# mm_router3.py
import os
import json
import argparse
from typing import List, Dict, Any

# 四个引擎都从这里 import（你刚刚写好的 engines.py）
try:
    from .engines import (
        run_vqa,
        run_chart_qa,
        run_table_text_llm,
        run_table_ocr_llm,
        run_generic_mm_llm,
        run_imgpair_vqa,
    )
except ImportError as e:
    raise ImportError(
        f"导入 engines 失败：{e}\n"
        f"请在当前目录下创建 engines.py，并实现 run_vqa / run_chart_qa / "
        f"run_table_text_llm / run_table_ocr_llm / run_generic_mm_llm 函数。"
    )


def decide_engine(item: Dict[str, Any]) -> str:
    """
    根据第二层的字段决定走哪个引擎：
      - ocrchart + table_source=text        -> table_text_llm
      - ocrchart + table_source=screenshot -> table_ocr_llm
      - datachart                          -> chart_qa
      - diagram / photo                    -> vqa
      - 其他                               -> generic_mm_llm
    """
    t = (item.get("type_final")
         or item.get("type")
         or "photo").lower()

    # Image-image consistency items (pair): use dedicated engine
    if (item.get("mm_mode") == "img_img") or (t == "imgpair"):
        return "imgpair_vqa"

    ocr_needed = bool(item.get("ocr_needed", False))
    table_source = (item.get("table_source") or "none").lower()

    if t == "ocrchart":
        if not ocr_needed and table_source == "text":
            return "table_text_llm"
        if ocr_needed and table_source == "screenshot":
            return "table_ocr_llm"
        # 奇怪情况，直接用兜底引擎
        return "generic_mm_llm"

    if t == "datachart":
        return "chart_qa"

    if t in ("diagram", "photo"):
        return "vqa"

    return "generic_mm_llm"


def run_engine(engine: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    真正调用对应 python 模块的地方。
    所有引擎都约定返回一个 dict（可以嵌套任何你想要的字段）。
    """
    if engine == "vqa":
        return run_vqa(item)
    if engine == "chart_qa":
        return run_chart_qa(item)
    if engine == "table_text_llm":
        return run_table_text_llm(item)
    if engine == "table_ocr_llm":
        return run_table_ocr_llm(item)
    if engine == "imgpair_vqa":
        return run_imgpair_vqa(item)

    # 兜底
    return run_generic_mm_llm(item)


def main():
    parser = argparse.ArgumentParser(
        description="MM 第三层：根据 all_routed.json 调用四个引擎并写回结果"
    )
    parser.add_argument(
        "--in_json",
        default=os.path.join("mm_router_routed", "all_routed.json"),
        help="第二层输出的 all_routed.json（默认: mm_router_routed/all_routed.json）",
    )
    parser.add_argument(
        "--out_json",
        default=os.path.join("mm_router_routed", "all_with_results.json"),
        help="第三层输出路径（默认: mm_router_routed/all_with_results.json）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.in_json):
        raise FileNotFoundError(args.in_json)

    with open(args.in_json, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    engine_stats: Dict[str, int] = {}

    for it in items:
        # 1) 决定走哪个引擎
        eng = decide_engine(it)
        it["engine"] = eng

        # 2) 调用对应引擎
        try:
            output = run_engine(eng, it)
            # 建议统一塞在 engine_output 里，里面可以随便设计结构
            it["engine_output"] = output
        except Exception as e:
            # 如果某个引擎报错，至少记录下来，不阻断整个 pipeline
            it["engine_error"] = str(e)

        engine_stats[eng] = engine_stats.get(eng, 0) + 1

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 打印一下分布
    print("Engine 使用统计：")
    for eng, cnt in sorted(engine_stats.items(), key=lambda x: x[0]):
        print(f"  {eng:15s}: {cnt:3d} items")
    print("=" * 40)
    print(f"总样本数: {len(items)}")
    print(f"输出已写入: {args.out_json}")


if __name__ == "__main__":
    main()
