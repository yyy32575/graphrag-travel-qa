"""检索器单元测试"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.indexing.faiss_index import FAISSIndex, ChunkResult
from src.indexing.bm25_index import BM25Index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker


def make_chunk_result(chunk_id: str, text: str = "测试文本", score: float = 0.8) -> ChunkResult:
    """创建测试用ChunkResult"""
    return ChunkResult(chunk_id=chunk_id, text=text, score=score)


class TestHybridRetriever:
    """HybridRetriever测试类"""

    def test_rrf_fusion(self) -> None:
        """测试RRF分数计算正确性"""
        dense_mock = MagicMock()
        sparse_mock = MagicMock()

        # 设置检索结果：chunk_a在密集检索排名1，chunk_b在稀疏检索排名1
        dense_mock.retrieve.return_value = [
            make_chunk_result("chunk_a", score=0.9),
            make_chunk_result("chunk_b", score=0.7),
        ]
        sparse_mock.retrieve.return_value = [
            make_chunk_result("chunk_b", score=0.8),
            make_chunk_result("chunk_a", score=0.6),
        ]

        retriever = HybridRetriever(dense_mock, sparse_mock, rrf_k=60)
        results = retriever.retrieve("测试查询", top_k=2)

        assert len(results) == 2
        # chunk_a: 1/(60+1) + 1/(60+2) ≈ 0.01639 + 0.01613 ≈ 0.03252
        # chunk_b: 1/(60+2) + 1/(60+1) ≈ 0.01613 + 0.01639 ≈ 0.03252
        # 两者RRF分数相等，顺序可能不同，但两者都应在结果中
        result_ids = {r.chunk_id for r in results}
        assert "chunk_a" in result_ids
        assert "chunk_b" in result_ids

    def test_hybrid_deduplication(self) -> None:
        """测试混合检索结果无重复chunk_id"""
        dense_mock = MagicMock()
        sparse_mock = MagicMock()

        # 两种检索器都返回相同的chunk
        dense_mock.retrieve.return_value = [
            make_chunk_result("chunk_001"),
            make_chunk_result("chunk_002"),
            make_chunk_result("chunk_003"),
        ]
        sparse_mock.retrieve.return_value = [
            make_chunk_result("chunk_001"),  # 重复
            make_chunk_result("chunk_004"),
            make_chunk_result("chunk_002"),  # 重复
        ]

        retriever = HybridRetriever(dense_mock, sparse_mock)
        results = retriever.retrieve("查询", top_k=10)

        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "存在重复的chunk_id"

    def test_rrf_higher_rank_gets_higher_score(self) -> None:
        """测试RRF排名越靠前分数越高"""
        dense_mock = MagicMock()
        sparse_mock = MagicMock()

        # chunk_top在两个检索器中都排名第1
        dense_mock.retrieve.return_value = [
            make_chunk_result("chunk_top"),
            make_chunk_result("chunk_mid"),
            make_chunk_result("chunk_bot"),
        ]
        sparse_mock.retrieve.return_value = [
            make_chunk_result("chunk_top"),
            make_chunk_result("chunk_mid"),
            make_chunk_result("chunk_bot"),
        ]

        retriever = HybridRetriever(dense_mock, sparse_mock, rrf_k=60)
        results = retriever.retrieve("查询", top_k=3)

        assert results[0].chunk_id == "chunk_top"
        assert results[0].score > results[1].score


class TestCrossEncoderReranker:
    """CrossEncoderReranker测试类"""

    @patch("src.retrieval.cross_encoder_reranker.CrossEncoder")
    def test_reranker_ordering(self, mock_ce_class: MagicMock) -> None:
        """测试重排序器改变结果顺序"""
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        # 模型给第二个候选更高分数
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.1])

        reranker = CrossEncoderReranker()
        candidates = [
            make_chunk_result("chunk_a", score=0.9),
            make_chunk_result("chunk_b", score=0.5),
            make_chunk_result("chunk_c", score=0.3),
        ]

        results = reranker.rerank("查询", candidates, top_k=3)

        # 应该按重排序后的分数排列，chunk_b得分最高
        assert results[0].chunk_id == "chunk_b"
        assert results[1].chunk_id == "chunk_a"
        assert results[2].chunk_id == "chunk_c"

    @patch("src.retrieval.cross_encoder_reranker.CrossEncoder")
    def test_reranker_empty_candidates(self, mock_ce_class: MagicMock) -> None:
        """测试空候选列表"""
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("查询", [], top_k=5)
        assert results == []


class TestDenseRetriever:
    """DenseRetriever测试类"""

    def test_dense_retriever_cache(self) -> None:
        """测试缓存机制：相同查询只调用一次索引"""
        mock_index = MagicMock()
        mock_index.search.return_value = [make_chunk_result("chunk_001")]

        retriever = DenseRetriever(mock_index)

        # 第一次调用
        results1 = retriever.retrieve("测试查询", top_k=5)
        # 第二次相同查询，应使用缓存
        results2 = retriever.retrieve("测试查询", top_k=5)

        # 索引只应被调用一次
        assert mock_index.search.call_count == 1
        assert results1 == results2

    def test_dense_retriever_cache_different_queries(self) -> None:
        """测试不同查询不使用缓存"""
        mock_index = MagicMock()
        mock_index.search.return_value = [make_chunk_result("chunk_001")]

        retriever = DenseRetriever(mock_index)
        retriever.retrieve("查询A", top_k=5)
        retriever.retrieve("查询B", top_k=5)

        assert mock_index.search.call_count == 2

    def test_dense_retriever_clear_cache(self) -> None:
        """测试清空缓存"""
        mock_index = MagicMock()
        mock_index.search.return_value = [make_chunk_result("chunk_001")]

        retriever = DenseRetriever(mock_index)
        retriever.retrieve("查询", top_k=5)
        retriever.clear_cache()
        retriever.retrieve("查询", top_k=5)

        assert mock_index.search.call_count == 2


class TestSparseRetriever:
    """SparseRetriever测试类"""

    def test_sparse_retriever_basic(self) -> None:
        """测试稀疏检索基本功能"""
        mock_index = MagicMock()
        expected_results = [
            make_chunk_result("chunk_001", score=0.9),
            make_chunk_result("chunk_002", score=0.7),
        ]
        mock_index.search.return_value = expected_results

        retriever = SparseRetriever(mock_index)
        results = retriever.retrieve("出行规则", top_k=5)

        assert results == expected_results
        mock_index.search.assert_called_once_with("出行规则", top_k=5)

    def test_sparse_retriever_returns_list(self) -> None:
        """测试稀疏检索返回列表类型"""
        mock_index = MagicMock()
        mock_index.search.return_value = []

        retriever = SparseRetriever(mock_index)
        results = retriever.retrieve("空结果查询", top_k=5)

        assert isinstance(results, list)
