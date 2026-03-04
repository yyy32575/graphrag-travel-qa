"""分块器单元测试"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.chunking.semantic_splitter import SemanticSplitter, Chunk
from src.chunking.overlap_window import OverlapWindowSplitter


class TestSemanticSplitter:
    """SemanticSplitter测试类"""

    @patch("src.chunking.semantic_splitter.SentenceTransformer")
    def test_semantic_splitter_basic(self, mock_st_class: MagicMock) -> None:
        """测试基本分块功能：不同嵌入向量应触发分块"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        # 使用正交向量确保低余弦相似度（相似度=0）
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        splitter = SemanticSplitter(similarity_threshold=0.9, min_chunk_size=1, max_chunk_size=10000)
        text = "第一句话内容。第二句话内容。第三句话内容。"
        chunks = splitter.split(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @patch("src.chunking.semantic_splitter.SentenceTransformer")
    def test_semantic_splitter_min_size(self, mock_st_class: MagicMock) -> None:
        """测试最小块大小约束"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        # 使用低相似度嵌入，强制尽量分块
        mock_model.encode.return_value = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        min_size = 50
        splitter = SemanticSplitter(
            similarity_threshold=0.9,
            min_chunk_size=min_size,
            max_chunk_size=10000,
        )
        text = "短句。短句。短句。短句。"
        chunks = splitter.split(text)

        # 所有块（除最后一块外）应满足最小大小
        for chunk in chunks[:-1]:
            assert len(chunk.text) >= min_size or len(chunks) == 1, \
                f"块 '{chunk.text}' 长度 {len(chunk.text)} 小于 {min_size}"

    @patch("src.chunking.semantic_splitter.SentenceTransformer")
    def test_semantic_splitter_max_size(self, mock_st_class: MagicMock) -> None:
        """测试最大块大小约束"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        # 使用高相似度嵌入，所有句子合并为一块
        mock_model.encode.return_value = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
            [1.0, 0.0],
        ], dtype=np.float32)

        max_size = 20
        splitter = SemanticSplitter(
            similarity_threshold=0.1,  # 非常低的阈值，几乎不分块
            min_chunk_size=1,
            max_chunk_size=max_size,
        )
        long_text = "这是一个比较长的句子用于测试。这是另一个比较长的句子。这是第三个比较长的句子用于测试最大大小。"
        chunks = splitter.split(long_text)

        for chunk in chunks:
            assert len(chunk.text) <= max_size, \
                f"块长度 {len(chunk.text)} 超过最大值 {max_size}"

    def test_overlap_window_char(self) -> None:
        """测试字符模式滑动窗口分块"""
        splitter = OverlapWindowSplitter(chunk_size=20, overlap_size=5, split_by="char")
        text = "这是一段用于测试滑动窗口分块功能的文本内容，包含足够多的字符来生成多个块。"
        chunks = splitter.split(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        # 验证块大小不超过chunk_size
        for chunk in chunks:
            assert len(chunk.text) <= 20

        # 验证重叠：除第一块外，相邻块应有重叠内容
        if len(chunks) >= 2:
            # 第一块的末尾5个字符应出现在第二块的开头
            overlap_text = chunks[0].text[-5:]
            assert overlap_text in chunks[1].text

    def test_overlap_window_token(self) -> None:
        """测试词（token）模式滑动窗口分块"""
        splitter = OverlapWindowSplitter(chunk_size=10, overlap_size=3, split_by="token")
        text = "这是一段用于测试分词滑动窗口分块功能的较长文本内容，包含多个词语。"
        chunks = splitter.split(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.get("split_by") == "token" for c in chunks)

    @patch("src.chunking.semantic_splitter.SentenceTransformer")
    def test_chunk_ids_unique(self, mock_st_class: MagicMock) -> None:
        """测试所有块ID唯一性"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        mock_model.encode.return_value = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        splitter = SemanticSplitter(
            similarity_threshold=0.9,
            min_chunk_size=1,
            max_chunk_size=10000,
        )
        text = "第一句内容测试。第二句内容不同。第三句内容测试。第四句内容不同。"
        chunks = splitter.split(text)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "存在重复的chunk_id"

    def test_overlap_window_invalid_params(self) -> None:
        """测试无效参数抛出异常"""
        with pytest.raises(ValueError):
            OverlapWindowSplitter(chunk_size=10, overlap_size=10)

        with pytest.raises(ValueError):
            OverlapWindowSplitter(split_by="word")

    @patch("src.chunking.semantic_splitter.SentenceTransformer")
    def test_empty_text(self, mock_st_class: MagicMock) -> None:
        """测试空文本输入"""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        splitter = SemanticSplitter()
        chunks = splitter.split("")
        assert chunks == []

        chunks = splitter.split("   ")
        assert chunks == []
