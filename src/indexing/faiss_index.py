"""FAISS向量索引模块 - 基于BGE模型的密集向量检索"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.semantic_splitter import Chunk


@dataclass
class ChunkResult:
    """检索结果数据类"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSIndex:
    """FAISS向量索引
    
    使用BGE嵌入模型构建FAISS内积索引（归一化后等同于余弦相似度），
    支持构建、保存、加载和检索操作。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        """初始化FAISS索引
        
        Args:
            model_name: 嵌入模型名称
            device: 计算设备
            batch_size: 编码批次大小
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []
        self._dim: int = 0

    def build(self, chunks: List[Chunk]) -> None:
        """从文本块列表构建FAISS索引
        
        Args:
            chunks: Chunk对象列表
        """
        if not chunks:
            raise ValueError("chunks列表不能为空")

        self.chunks = list(chunks)
        texts = [c.text for c in chunks]

        # 批量编码并归一化
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        self._dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        """保存索引到磁盘
        
        Args:
            path: 保存路径（不含扩展名），将保存 {path}.faiss 和 {path}.pkl
        """
        if self.index is None:
            raise RuntimeError("索引尚未构建，请先调用 build()")

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "dim": self._dim,
            }, f)

    def load(self, path: str) -> None:
        """从磁盘加载索引
        
        Args:
            path: 加载路径（不含扩展名），读取 {path}.faiss 和 {path}.pkl
        """
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._dim = data["dim"]

    def search(self, query: str, top_k: int = 10) -> List[ChunkResult]:
        """检索与查询最相似的文本块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            ChunkResult对象列表，按相似度降序排列
        """
        if self.index is None:
            raise RuntimeError("索引尚未构建或加载")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results: List[ChunkResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(ChunkResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                metadata=dict(chunk.metadata),
            ))
        return results
