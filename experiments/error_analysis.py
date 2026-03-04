"""错误分析实验"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 错误类别定义
ERROR_CATEGORIES = {
    "no_evidence": "未找到相关证据",
    "wrong_evidence": "检索到错误证据",
    "generation_error": "生成过程出错",
    "low_confidence": "置信度过低",
    "correct": "回答正确",
}


def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """加载评测查询"""
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def categorize_error(
    query: str,
    predicted: str,
    reference: str,
    retrieval_count: int,
    confidence: float,
) -> str:
    """分类错误类型
    
    Args:
        query: 查询文本
        predicted: 预测答案
        reference: 参考答案
        retrieval_count: 检索到的结果数量
        confidence: 置信度分数
        
    Returns:
        错误类别
    """
    if not predicted or len(predicted) < 5:
        return "generation_error"
    
    if retrieval_count == 0:
        return "no_evidence"
    
    if confidence < 0.3:
        return "low_confidence"
    
    # 检查答案是否相关（简单关键词重叠）
    ref_chars = set(reference[:50])
    pred_chars = set(predicted[:100])
    overlap = len(ref_chars & pred_chars) / len(ref_chars) if ref_chars else 0
    
    if overlap < 0.3:
        return "wrong_evidence"
    
    return "correct"


def run_error_analysis(
    queries: List[Dict[str, Any]],
    data_path: str,
) -> List[Dict[str, Any]]:
    """运行错误分析
    
    Args:
        queries: 查询列表
        data_path: 数据文件路径
        
    Returns:
        每条查询的分析结果
    """
    from src.chunking.overlap_window import OverlapWindowSplitter
    from src.indexing.bm25_index import BM25Index
    from src.retrieval.sparse_retriever import SparseRetriever

    # 读取数据分块
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = OverlapWindowSplitter(chunk_size=512, overlap_size=64)
    chunks = splitter.split(text)

    bm25_index = BM25Index()
    bm25_index.build(chunks)
    retriever = SparseRetriever(bm25_index)

    analysis_results = []

    for i, query_item in enumerate(queries):
        query = query_item.get("query", "")
        reference = query_item.get("answer", "")
        difficulty = query_item.get("difficulty", "unknown")
        tags = query_item.get("tags", [])

        # 检索
        results = retriever.retrieve(query, top_k=5)
        retrieval_count = len(results)
        avg_score = sum(r.score for r in results) / retrieval_count if retrieval_count else 0

        # 用检索结果的最佳匹配作为预测
        predicted = results[0].text[:200] if results else ""
        confidence = results[0].score if results else 0.0

        # 分类错误
        error_category = categorize_error(
            query, predicted, reference, retrieval_count, confidence
        )

        analysis_results.append({
            "query_id": i,
            "query": query[:100],
            "reference": reference[:100],
            "predicted": predicted[:100],
            "difficulty": difficulty,
            "tags": "|".join(tags) if isinstance(tags, list) else str(tags),
            "retrieval_count": retrieval_count,
            "avg_retrieval_score": round(avg_score, 4),
            "confidence": round(confidence, 4),
            "error_category": error_category,
            "error_description": ERROR_CATEGORIES.get(error_category, "未知"),
        })

    return analysis_results


def print_error_summary(results: List[Dict[str, Any]]) -> None:
    """打印错误分析摘要"""
    from collections import Counter

    error_counts = Counter(r["error_category"] for r in results)
    total = len(results)

    print("\n" + "=" * 50)
    print("错误分析摘要")
    print("=" * 50)
    for category, description in ERROR_CATEGORIES.items():
        count = error_counts.get(category, 0)
        pct = count / total * 100 if total else 0
        print(f"  {description:<20}: {count:>4} ({pct:>5.1f}%)")
    print(f"  {'总计':<20}: {total:>4}")
    print("=" * 50)


def main() -> None:
    """执行错误分析"""
    data_path = "data/raw/sample_travel_rules.md"
    output_dir = "experiments/results"
    output_file = os.path.join(output_dir, "error_analysis.csv")
    os.makedirs(output_dir, exist_ok=True)

    # 加载查询
    queries = []
    for qf in ["data/eval/general_queries.jsonl", "data/eval/complex_queries.jsonl"]:
        try:
            queries.extend(load_queries(qf))
        except FileNotFoundError:
            print(f"警告：未找到 {qf}")

    if not queries:
        queries = [
            {"query": "行李限额", "answer": "20公斤", "difficulty": "simple", "tags": ["aviation"]},
            {"query": "退票规定", "answer": "手续费", "difficulty": "simple", "tags": ["aviation"]},
        ]

    print(f"分析 {len(queries)} 条查询...")

    try:
        results = run_error_analysis(queries, data_path)
    except Exception as e:
        print(f"分析出错：{e}")
        results = []

    if not results:
        print("无分析结果")
        return

    print_error_summary(results)

    # 保存CSV
    fieldnames = [
        "query_id", "query", "reference", "predicted",
        "difficulty", "tags", "retrieval_count",
        "avg_retrieval_score", "confidence",
        "error_category", "error_description",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n分析结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
