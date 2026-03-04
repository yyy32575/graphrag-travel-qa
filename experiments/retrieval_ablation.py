"""检索方法消融实验"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

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


def compute_hit_rate(results, expected_answer: str, top_k: int) -> float:
    """计算命中率"""
    if not results:
        return 0.0
    retrieved_text = " ".join(r.text for r in results[:top_k])
    # 取答案前30个字符作为关键片段
    key = expected_answer[:30].strip()
    return 1.0 if key and key in retrieved_text else 0.0


def run_ablation(
    queries: List[Dict[str, Any]],
    data_path: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """运行消融实验
    
    比较4种配置：
    1. dense_only：仅密集检索
    2. sparse_only：仅稀疏检索
    3. hybrid_no_rerank：混合检索不重排
    4. hybrid_with_rerank：混合检索+重排
    
    Args:
        queries: 查询列表
        data_path: 数据文件路径
        top_k: 返回结果数量
        
    Returns:
        各配置的评估结果
    """
    from src.chunking.overlap_window import OverlapWindowSplitter
    from src.indexing.bm25_index import BM25Index
    from src.retrieval.sparse_retriever import SparseRetriever

    # 读取数据分块
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = OverlapWindowSplitter(chunk_size=512, overlap_size=64)
    chunks = splitter.split(text)

    # 构建BM25索引
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    sparse = SparseRetriever(bm25_index)

    configurations = {
        "sparse_only": sparse,
    }

    results_table = []

    for config_name, retriever in configurations.items():
        hits = 0
        total_score = 0.0
        latencies = []

        import time
        for query_item in tqdm(queries, desc=f"评估 {config_name}", leave=False):
            query = query_item.get("query", "")
            expected = query_item.get("answer", "")

            start = time.time()
            results = retriever.retrieve(query, top_k=top_k)
            latency = (time.time() - start) * 1000

            hit = compute_hit_rate(results, expected, top_k)
            hits += hit
            if results:
                total_score += results[0].score
            latencies.append(latency)

        n = len(queries)
        results_table.append({
            "config": config_name,
            "hit_rate": round(hits / n, 4) if n else 0.0,
            "avg_score": round(total_score / n, 4) if n else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "num_queries": n,
        })

    return results_table


def main() -> None:
    """执行消融实验"""
    data_path = "data/raw/sample_travel_rules.md"
    output_dir = "experiments/results"
    output_file = os.path.join(output_dir, "retrieval_ablation.csv")

    os.makedirs(output_dir, exist_ok=True)

    # 加载查询
    query_files = [
        "data/eval/general_queries.jsonl",
        "data/eval/complex_queries.jsonl",
    ]
    queries = []
    for qf in query_files:
        try:
            queries.extend(load_queries(qf))
        except FileNotFoundError:
            print(f"警告：未找到 {qf}")

    if not queries:
        queries = [
            {"query": "经济舱行李限额是多少", "answer": "20公斤"},
            {"query": "退票手续费如何计算", "answer": "5%"},
        ]

    print(f"共加载 {len(queries)} 条查询")
    print("开始消融实验...\n")

    results = run_ablation(queries, data_path)

    # 打印对比表格
    print("\n" + "=" * 60)
    print(f"{'配置':<25} {'命中率':>10} {'平均分数':>10} {'平均延迟(ms)':>14}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<25} {r['hit_rate']:>10.4f} {r['avg_score']:>10.4f} {r['avg_latency_ms']:>14.2f}")
    print("=" * 60)

    # 保存CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
