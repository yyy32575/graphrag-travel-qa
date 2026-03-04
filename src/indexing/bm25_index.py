"""BM25稀疏索引模块 - 基于BM25算法的关键词检索"""
from __future__ import annotations

import pickle
from typing import List

import jieba
from rank_bm25 import BM25Okapi

from src.chunking.semantic_splitter import Chunk
from src.indexing.faiss_index import ChunkResult


class BM25Index:
    """BM25稀疏索引
    
    使用jieba中文分词和BM25Okapi算法构建稀疏检索索引，
    BM25分数归一化到[0,1]区间。
    """

    def __init__(self) -> None:
        """初始化BM25索引"""
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """使用jieba进行中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            词语列表
        """
        return list(jieba.cut(text))

    def build(self, chunks: List[Chunk]) -> None:
        """从文本块列表构建BM25索引
        
        Args:
            chunks: Chunk对象列表
        """
        if not chunks:
            raise ValueError("chunks列表不能为空")

        self.chunks = list(chunks)
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def save(self, path: str) -> None:
        """保存索引到磁盘
        
        Args:
            path: 保存路径（pickle格式）
        """
        if self.bm25 is None:
            raise RuntimeError("索引尚未构建，请先调用 build()")

        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "tokenized_corpus": self._tokenized_corpus,
            }, f)

    def load(self, path: str) -> None:
        """从磁盘加载索引
        
        Args:
            path: 加载路径
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[ChunkResult]:
        """检索与查询最相关的文本块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            ChunkResult对象列表，按BM25分数降序排列
        """
        if self.bm25 is None:
            raise RuntimeError("索引尚未构建或加载")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 归一化分数到[0,1]
        max_score = float(scores.max()) if scores.max() > 0 else 1.0
        normalized_scores = scores / max_score

        # 获取top_k索引
        k = min(top_k, len(self.chunks))
        top_indices = scores.argsort()[::-1][:k]

        results: List[ChunkResult] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(ChunkResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(normalized_scores[idx]),
                metadata=dict(chunk.metadata),
            ))
        return results
