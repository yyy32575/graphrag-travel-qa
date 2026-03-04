"""混合检索器模块 - 使用RRF融合密集和稀疏检索结果"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.indexing.faiss_index import ChunkResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    """混合检索器
    
    使用互惠排名融合（RRF）算法合并密集检索和稀疏检索的结果。
    RRF公式：score(d) = Σ 1 / (k + rank(d))
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> None:
        """初始化混合检索器
        
        Args:
            dense_retriever: 密集检索器
            sparse_retriever: 稀疏检索器
            rrf_k: RRF算法中的k参数，默认60
            dense_top_k: 密集检索返回数量
            sparse_top_k: 稀疏检索返回数量
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k

    def _rrf_score(self, rank: int) -> float:
        """计算单个排名的RRF分数
        
        Args:
            rank: 排名（从1开始）
            
        Returns:
            RRF分数
        """
        return 1.0 / (self.rrf_k + rank)

    def retrieve(self, query: str, top_k: int = 10) -> List[ChunkResult]:
        """使用RRF融合检索结果
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            按RRF分数降序排列的ChunkResult列表（无重复）
        """
        # 分别获取密集和稀疏检索结果
        dense_results = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)

        # 计算RRF分数
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, ChunkResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] += self._rrf_score(rank)
            chunk_map[result.chunk_id] = result

        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.chunk_id] += self._rrf_score(rank)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result

        # 按RRF分数降序排列
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[:top_k]:
            result = chunk_map[chunk_id]
            # 用RRF分数替换原始分数
            results.append(ChunkResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=rrf_scores[chunk_id],
                metadata=result.metadata,
            ))
        return results
