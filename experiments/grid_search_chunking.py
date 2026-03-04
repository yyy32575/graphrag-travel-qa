"""分块参数网格搜索实验"""
from __future__ import annotations

import csv
import json
import os
import sys
from itertools import product
from typing import Any, Dict, List

from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """加载评测查询
    
    Args:
        query_file: JSONL格式的查询文件路径
        
    Returns:
        查询字典列表
    """
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def evaluate_config(
    chunk_size: int,
    overlap: int,
    top_k: int,
    queries: List[Dict[str, Any]],
    data_path: str,
) -> Dict[str, Any]:
    """评估单个参数配置
    
    Args:
        chunk_size: 块大小
        overlap: 重叠大小
        top_k: 返回结果数量
        queries: 查询列表
        data_path: 数据文件路径
        
    Returns:
        评估结果字典
    """
    from src.chunking.overlap_window import OverlapWindowSplitter
    from src.indexing.faiss_index import FAISSIndex
    from src.indexing.bm25_index import BM25Index
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    try:
        # 读取数据并分块
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = OverlapWindowSplitter(
            chunk_size=chunk_size,
            overlap_size=overlap,
            split_by="char",
        )
        chunks = splitter.split(text)

        if not chunks:
            return {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                "num_chunks": 0,
                "hit_rate": 0.0,
                "avg_score": 0.0,
                "error": "no_chunks",
            }

        # 构建BM25索引（避免模型下载）
        bm25_index = BM25Index()
        bm25_index.build(chunks)
        retriever = SparseRetriever(bm25_index)

        # 评估命中率
        hits = 0
        total_score = 0.0

        for query_item in queries:
            query = query_item.get("query", "")
            expected_answer = query_item.get("answer", "")

            results = retriever.retrieve(query, top_k=top_k)

            # 简单启发式：检查答案关键词是否在检索结果中
            answer_keywords = set(expected_answer[:50].split())
            retrieved_text = " ".join(r.text[:100] for r in results)
            
            keyword_hits = sum(1 for kw in answer_keywords if kw in retrieved_text)
            hit = keyword_hits > 0
            
            if hit:
                hits += 1
            if results:
                total_score += results[0].score

        hit_rate = hits / len(queries) if queries else 0.0
        avg_score = total_score / len(queries) if queries else 0.0

        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "num_chunks": len(chunks),
            "hit_rate": round(hit_rate, 4),
            "avg_score": round(avg_score, 4),
            "error": "",
        }
    except Exception as e:
        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "num_chunks": 0,
            "hit_rate": 0.0,
            "avg_score": 0.0,
            "error": str(e),
        }


def main() -> None:
    """执行网格搜索实验"""
    # 参数网格
    chunk_sizes = [256, 512, 1024]
    overlaps = [0, 64, 128]
    top_ks = [3, 5, 10]

    # 数据路径
    data_path = "data/raw/sample_travel_rules.md"
    query_file = "data/eval/general_queries.jsonl"
    output_dir = "experiments/results"
    output_file = os.path.join(output_dir, "grid_search_results.csv")

    os.makedirs(output_dir, exist_ok=True)

    print("加载评测查询...")
    try:
        queries = load_queries(query_file)
        print(f"已加载 {len(queries)} 条查询")
    except FileNotFoundError:
        print(f"警告：未找到查询文件 {query_file}，使用示例查询")
        queries = [
            {"query": "经济舱行李限额", "answer": "20公斤"},
            {"query": "退票手续费", "answer": "5%"},
        ]

    # 生成所有参数组合
    param_combos = list(product(chunk_sizes, overlaps, top_ks))
    # 过滤无效组合（overlap >= chunk_size）
    param_combos = [(cs, ov, tk) for cs, ov, tk in param_combos if ov < cs]

    print(f"共 {len(param_combos)} 种参数组合")

    results = []
    fieldnames = ["chunk_size", "overlap", "top_k", "num_chunks", "hit_rate", "avg_score", "error"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for chunk_size, overlap, top_k in tqdm(param_combos, desc="网格搜索"):
            result = evaluate_config(chunk_size, overlap, top_k, queries, data_path)
            results.append(result)
            writer.writerow(result)
            csvfile.flush()

    print(f"\n实验完成！结果已保存到 {output_file}")

    # 打印最佳配置
    if results:
        best = max(results, key=lambda x: x["hit_rate"])
        print(f"\n最佳配置（命中率）：")
        print(f"  chunk_size={best['chunk_size']}, overlap={best['overlap']}, top_k={best['top_k']}")
        print(f"  命中率={best['hit_rate']:.4f}, 平均分数={best['avg_score']:.4f}")


if __name__ == "__main__":
    main()
