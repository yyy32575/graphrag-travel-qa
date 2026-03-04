"""滑动窗口分块器模块 - 基于固定窗口和重叠的文本分块"""
from __future__ import annotations

from typing import List

from .semantic_splitter import Chunk


class OverlapWindowSplitter:
    """基于滑动窗口的文本分块器
    
    使用固定窗口大小和重叠区域进行文本分块，
    支持按字符或按词（token）分块。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 64,
        split_by: str = "char",
    ) -> None:
        """初始化滑动窗口分块器
        
        Args:
            chunk_size: 每块的大小（字符数或词数）
            overlap_size: 相邻块的重叠大小
            split_by: 分割方式，'char'（字符）或 'token'（词）
        """
        if split_by not in ("char", "token"):
            raise ValueError(f"split_by must be 'char' or 'token', got '{split_by}'")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.split_by = split_by

    def _tokenize(self, text: str) -> List[str]:
        """简单的中文分词（按字符分割）
        
        Args:
            text: 输入文本
            
        Returns:
            词语列表
        """
        # 简单按空格和标点分割，适用于中文
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\s]', text)
        return tokens

    def split(self, text: str) -> List[Chunk]:
        """将文本按滑动窗口分割
        
        Args:
            text: 输入文本
            
        Returns:
            Chunk对象列表
        """
        if not text.strip():
            return []

        if self.split_by == "char":
            return self._split_by_char(text)
        else:
            return self._split_by_token(text)

    def _split_by_char(self, text: str) -> List[Chunk]:
        """按字符数进行滑动窗口分块
        
        Args:
            text: 输入文本
            
        Returns:
            Chunk对象列表
        """
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap_size
        start = 0
        i = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"chunk_{i:04d}",
                    text=chunk_text,
                    metadata={
                        "source": "overlap_window",
                        "split_by": "char",
                        "start_char": start,
                        "end_char": end,
                        "chunk_index": i,
                    }
                ))
                i += 1
            start += step
            if end == len(text):
                break

        return chunks

    def _split_by_token(self, text: str) -> List[Chunk]:
        """按词数进行滑动窗口分块
        
        Args:
            text: 输入文本
            
        Returns:
            Chunk对象列表
        """
        tokens = self._tokenize(text)
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap_size
        start = 0
        i = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = "".join(chunk_tokens)
            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"chunk_{i:04d}",
                    text=chunk_text,
                    metadata={
                        "source": "overlap_window",
                        "split_by": "token",
                        "start_token": start,
                        "end_token": end,
                        "token_count": len(chunk_tokens),
                        "chunk_index": i,
                    }
                ))
                i += 1
            start += step
            if end == len(tokens):
                break

        return chunks
