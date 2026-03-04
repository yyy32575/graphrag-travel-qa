"""语义分块器模块 - 基于语义相似度的文本分块"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """文本块数据类"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticSplitter:
    """基于语义相似度的文本分块器
    
    使用句子嵌入模型计算相邻句子间的余弦相似度，
    当相似度低于阈值时进行分块，同时遵守最小/最大块大小约束。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        """初始化语义分块器
        
        Args:
            model_name: 句子嵌入模型名称
            similarity_threshold: 分块阈值，低于此相似度则分块
            min_chunk_size: 最小块大小（字符数）
            max_chunk_size: 最大块大小（字符数）
            device: 计算设备，'cpu' 或 'cuda'
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _sentence_tokenize(self, text: str) -> List[str]:
        """中文句子分割
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 按中文句号、感叹号、问号分割，保留分隔符
        sentences = re.split(r'(?<=[。！？\.\!\?])', text)
        # 过滤空白句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            v1: 向量1
            v2: 向量2
            
        Returns:
            余弦相似度值
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def split(self, text: str) -> List[Chunk]:
        """将文本分割为语义连贯的块
        
        Args:
            text: 输入文本
            
        Returns:
            Chunk对象列表
        """
        if not text.strip():
            return []

        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []

        # 如果只有一个句子或文本很短，直接返回
        if len(sentences) == 1 or len(text) <= self.min_chunk_size:
            return [Chunk(
                chunk_id="chunk_0000",
                text=text.strip(),
                metadata={"source": "semantic_splitter", "sentence_count": len(sentences)}
            )]

        # 计算所有句子的嵌入向量
        embeddings = self.model.encode(sentences)

        # 基于相似度找分割点
        split_points: List[int] = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.similarity_threshold:
                split_points.append(i + 1)

        # 根据分割点构建初始块
        chunks_text: List[str] = []
        current_start = 0
        for split_point in split_points:
            chunk_text = "".join(sentences[current_start:split_point])
            if chunk_text.strip():
                chunks_text.append(chunk_text.strip())
            current_start = split_point
        # 添加最后一块
        last_chunk = "".join(sentences[current_start:])
        if last_chunk.strip():
            chunks_text.append(last_chunk.strip())

        # 合并过小的块，分割过大的块
        final_chunks_text = self._enforce_size_constraints(chunks_text)

        # 转换为Chunk对象
        chunks = []
        for i, chunk_text in enumerate(final_chunks_text):
            chunks.append(Chunk(
                chunk_id=f"chunk_{i:04d}",
                text=chunk_text,
                metadata={
                    "source": "semantic_splitter",
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                }
            ))
        return chunks

    def _enforce_size_constraints(self, chunks: List[str]) -> List[str]:
        """强制执行最小/最大块大小约束
        
        Args:
            chunks: 初始文本块列表
            
        Returns:
            满足大小约束的文本块列表
        """
        if not chunks:
            return []

        result: List[str] = []
        buffer = ""

        for chunk in chunks:
            if not buffer:
                buffer = chunk
            elif len(buffer) < self.min_chunk_size:
                # buffer hasn't reached minimum size yet — keep accumulating
                buffer = buffer + chunk
            else:
                # buffer meets minimum size — commit it and start fresh
                result.append(buffer)
                buffer = chunk

            # 分割过大的块
            while len(buffer) > self.max_chunk_size:
                result.append(buffer[:self.max_chunk_size])
                buffer = buffer[self.max_chunk_size:]

        # 处理剩余的buffer
        if buffer:
            if result and len(buffer) < self.min_chunk_size:
                # 将剩余内容合并到最后一块
                result[-1] = result[-1] + buffer
                # 如果合并后超过最大大小，再次分割
                if len(result[-1]) > self.max_chunk_size:
                    last = result.pop()
                    while len(last) > self.max_chunk_size:
                        result.append(last[:self.max_chunk_size])
                        last = last[self.max_chunk_size:]
                    if last:
                        result.append(last)
            else:
                result.append(buffer)

        return [c for c in result if c.strip()]
