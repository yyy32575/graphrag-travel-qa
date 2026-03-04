"""按难度分桶评估实验"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """加载评测查询"""
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算评估指标
    
    Args:
        predictions: 预测答案列表
        references: 参考答案列表
        
    Returns:
        包含precision, recall, f1, exact_match的指标字典
    """
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

    exact_matches = 0
    precisions = []
    recalls = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(jieba.cut(pred.strip()))
        ref_tokens = set(jieba.cut(ref.strip()))

        # 精确匹配
        if pred.strip() == ref.strip():
            exact_matches += 1

        # 字符级别的precision和recall
        if not pred_tokens and not ref_tokens:
            precisions.append(1.0)
            recalls.append(1.0)
        elif not pred_tokens:
            precisions.append(0.0)
            recalls.append(0.0)
        elif not ref_tokens:
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            precisions.append(precision)
            recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)
          if (avg_precision + avg_recall) > 0 else 0.0)
    exact_match_rate = exact_matches / len(predictions)

    return {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(exact_match_rate, 4),
    }


def simulate_retrieval(query: str, chunks_text: str, top_k: int = 3) -> str:
    """简单的关键词匹配模拟检索（用于无模型评估）"""
    query_words = set(query.replace("？", "").replace("?", "").split())
    
    # 返回包含最多查询词的文本片段
    best_match = ""
    best_count = 0
    
    for line in chunks_text.split("\n"):
        count = sum(1 for w in query_words if w in line)
        if count > best_count:
            best_count = count
            best_match = line
    
    return best_match[:200] if best_match else ""


def main() -> None:
    """执行分桶评估"""
    output_dir = "experiments/results"
    output_file = os.path.join(output_dir, "eval_bucketed.csv")
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有查询
    all_queries: List[Dict[str, Any]] = []
    for qf in ["data/eval/general_queries.jsonl", "data/eval/complex_queries.jsonl"]:
        try:
            all_queries.extend(load_queries(qf))
        except FileNotFoundError:
            print(f"警告：未找到 {qf}")

    if not all_queries:
        print("未找到评测数据，使用示例数据")
        all_queries = [
            {"query": "行李限额", "answer": "20公斤", "difficulty": "simple", "tags": ["aviation"]},
            {"query": "多跳推理查询", "answer": "复杂答案", "difficulty": "multi_hop", "tags": ["aviation", "visa"]},
        ]

    # 尝试加载数据文本
    data_text = ""
    try:
        with open("data/raw/sample_travel_rules.md", "r", encoding="utf-8") as f:
            data_text = f.read()
    except FileNotFoundError:
        pass

    # 按难度分组
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for q in all_queries:
        difficulty = q.get("difficulty", "unknown")
        buckets[difficulty].append(q)

    print(f"共 {len(all_queries)} 条查询，分布：")
    for diff, qs in buckets.items():
        print(f"  {diff}: {len(qs)} 条")

    # 对每个分桶计算指标
    bucket_results = []
    all_predictions = []
    all_references = []

    for difficulty, queries in buckets.items():
        predictions = []
        references = []

        for q in queries:
            query_text = q.get("query", "")
            reference = q.get("answer", "")

            # 使用简单模拟检索生成预测
            prediction = simulate_retrieval(query_text, data_text)
            predictions.append(prediction)
            references.append(reference)

        metrics = compute_metrics(predictions, references)
        bucket_result = {
            "difficulty": difficulty,
            "count": len(queries),
            **metrics,
        }
        bucket_results.append(bucket_result)
        all_predictions.extend(predictions)
        all_references.extend(references)

    # 整体指标
    overall = compute_metrics(all_predictions, all_references)
    bucket_results.append({
        "difficulty": "overall",
        "count": len(all_queries),
        **overall,
    })

    # 打印结果表格
    print("\n" + "=" * 70)
    print(f"{'难度级别':<15} {'数量':>6} {'精确率':>10} {'召回率':>10} {'F1':>10} {'精确匹配':>10}")
    print("-" * 70)
    for r in bucket_results:
        print(
            f"{r['difficulty']:<15} {r['count']:>6} "
            f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1']:>10.4f} {r['exact_match']:>10.4f}"
        )
    print("=" * 70)

    # 保存CSV
    fieldnames = ["difficulty", "count", "precision", "recall", "f1", "exact_match"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bucket_results)

    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
