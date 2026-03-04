"""运行评估脚本
使用方法：python scripts/run_eval.py [--eval-file PATH] [--output-dir DIR] [--top-k N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_queries(eval_file: str) -> List[Dict[str, Any]]:
    """加载评测查询"""
    queries = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算评估指标"""
    if not results:
        return {"hit_rate": 0.0, "avg_confidence": 0.0, "avg_latency_ms": 0.0}

    correct = sum(1 for r in results if r.get("is_hit", False))
    avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results)

    return {
        "hit_rate": round(correct / len(results), 4),
        "avg_confidence": round(avg_confidence, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }


def main() -> None:
    """主函数：执行评估"""
    parser = argparse.ArgumentParser(
        description="运行GraphRAG系统评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/general_queries.jsonl",
        help="评测数据文件路径（JSONL格式）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索返回结果数量",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed",
        help="索引目录",
    )

    args = parser.parse_args()

    print(f"GraphRAG出行知识问答系统 - 评估工具")
    print(f"=" * 50)

    # 加载查询
    if not os.path.exists(args.eval_file):
        print(f"错误：评测文件不存在：{args.eval_file}")
        sys.exit(1)

    queries = load_queries(args.eval_file)
    print(f"已加载 {len(queries)} 条评测查询")

    # 初始化流水线
    print("初始化流水线...")
    try:
        from src.pipeline.rag_pipeline import GraphRAGPipeline
        pipeline = GraphRAGPipeline(config_path=args.config)
    except Exception as e:
        print(f"初始化失败：{e}")
        sys.exit(1)

    # 加载索引
    faiss_path = os.path.join(args.index_dir, "faiss_index.faiss")
    if os.path.exists(faiss_path):
        print("加载索引...")
        try:
            pipeline.load_indices(args.index_dir)
        except Exception as e:
            print(f"索引加载失败：{e}")
            sys.exit(1)
    else:
        print(f"未找到索引文件，请先运行 scripts/build_index.py")
        sys.exit(1)

    # 运行评估
    print(f"\n开始评估（共 {len(queries)} 条查询）...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "eval_results.csv")

    eval_results = []
    fieldnames = [
        "query_id", "query", "reference", "predicted_answer",
        "confidence", "latency_ms", "cited_chunks", "is_hit", "difficulty",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, query_item in enumerate(queries):
            query = query_item.get("query", "")
            reference = query_item.get("answer", "")
            difficulty = query_item.get("difficulty", "unknown")

            try:
                result = pipeline.query(query, top_k=args.top_k)

                # 简单命中判断：答案关键词是否在预测答案中
                ref_key = reference[:30].strip()
                is_hit = bool(ref_key) and ref_key in result.answer

                row = {
                    "query_id": i,
                    "query": query,
                    "reference": reference[:200],
                    "predicted_answer": result.answer[:200],
                    "confidence": round(result.confidence, 4),
                    "latency_ms": round(result.latency_ms, 2),
                    "cited_chunks": "|".join(result.stage_details.get("cited_chunks", [])),
                    "is_hit": is_hit,
                    "difficulty": difficulty,
                }
            except Exception as e:
                row = {
                    "query_id": i,
                    "query": query,
                    "reference": reference[:200],
                    "predicted_answer": f"ERROR: {e}",
                    "confidence": 0.0,
                    "latency_ms": 0.0,
                    "cited_chunks": "",
                    "is_hit": False,
                    "difficulty": difficulty,
                }

            eval_results.append(row)
            writer.writerow(row)
            csvfile.flush()

            if (i + 1) % 10 == 0:
                print(f"  进度：{i+1}/{len(queries)}")

    # 计算并打印指标
    metrics = compute_metrics(eval_results)
    print(f"\n评估完成！")
    print(f"  命中率：{metrics['hit_rate']:.4f}")
    print(f"  平均置信度：{metrics['avg_confidence']:.4f}")
    print(f"  平均延迟：{metrics['avg_latency_ms']:.2f}ms")
    print(f"  结果已保存到：{output_file}")


if __name__ == "__main__":
    main()
