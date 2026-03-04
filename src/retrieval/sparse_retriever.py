"""稀疏检索器模块 - 封装BM25索引"""
from __future__ import annotations

from typing import List

from src.indexing.bm25_index import BM25Index
from src.indexing.faiss_index import ChunkResult


class SparseRetriever:
    """稀疏BM25检索器
    
    封装BM25Index，提供基于关键词的稀疏检索功能。
    """

    def __init__(self, bm25_index: BM25Index) -> None:
        """初始化稀疏检索器
        
        Args:
            bm25_index: 已构建或加载的BM25索引
        """
        self._index = bm25_index

    def retrieve(self, query: str, top_k: int = 10) -> List[ChunkResult]:
        """检索与查询最相关的文本块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            ChunkResult对象列表，按BM25分数降序排列
        """
        return self._index.search(query, top_k=top_k)
