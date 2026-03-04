"""密集检索器模块 - 封装FAISS向量索引并带查询缓存"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.indexing.faiss_index import FAISSIndex, ChunkResult


class DenseRetriever:
    """密集向量检索器
    
    封装FAISSIndex，提供带缓存的语义检索功能。
    使用字典缓存查询结果，避免重复计算。
    """

    def __init__(self, faiss_index: FAISSIndex) -> None:
        """初始化密集检索器
        
        Args:
            faiss_index: 已构建或加载的FAISS索引
        """
        self._index = faiss_index
        self._cache: Dict[Tuple[str, int], List[ChunkResult]] = {}

    def retrieve(self, query: str, top_k: int = 10) -> List[ChunkResult]:
        """检索与查询最相似的文本块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            ChunkResult对象列表，按相似度降序排列
        """
        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = self._index.search(query, top_k=top_k)
        self._cache[cache_key] = results
        return results

    def clear_cache(self) -> None:
        """清空查询缓存"""
        self._cache.clear()
