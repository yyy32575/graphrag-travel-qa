"""流水线集成测试"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.indexing.faiss_index import ChunkResult
from src.generation.evidence_constrained import EvidenceConstrainedGenerator, GenerationResult
from src.generation.confidence_gate import ConfidenceGate, GateResult


def make_chunk_result(chunk_id: str, score: float = 0.8) -> ChunkResult:
    """创建测试用ChunkResult"""
    return ChunkResult(chunk_id=chunk_id, text=f"测试文本 {chunk_id}", score=score, metadata={})


class TestEvidenceConstrainedGenerator:
    """EvidenceConstrainedGenerator测试类"""

    @patch("src.generation.evidence_constrained.ChatOpenAI")
    def test_generation_result_parsing_valid_json(self, mock_chat_class: MagicMock) -> None:
        """测试有效JSON响应解析"""
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        response_json = json.dumps({
            "answer": "经济舱免费托运20公斤行李。",
            "cited_chunks": ["chunk_0001", "chunk_0002"],
            "confidence_score": 0.9,
        })
        mock_response = MagicMock()
        mock_response.content = response_json
        mock_llm.invoke.return_value = mock_response

        generator = EvidenceConstrainedGenerator()
        evidence = [make_chunk_result("chunk_0001"), make_chunk_result("chunk_0002")]
        result = generator.generate("经济舱行李规定？", evidence)

        assert result.answer == "经济舱免费托运20公斤行李。"
        assert "chunk_0001" in result.cited_chunks
        assert result.confidence_score == 0.9

    @patch("src.generation.evidence_constrained.ChatOpenAI")
    def test_generation_result_parsing_invalid_json(self, mock_chat_class: MagicMock) -> None:
        """测试无效JSON响应的降级处理"""
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        mock_response = MagicMock()
        mock_response.content = "这是一个非JSON的纯文本回答"
        mock_llm.invoke.return_value = mock_response

        generator = EvidenceConstrainedGenerator()
        evidence = [make_chunk_result("chunk_0001")]
        result = generator.generate("查询", evidence)

        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    @patch("src.generation.evidence_constrained.ChatOpenAI")
    def test_generation_empty_evidence(self, mock_chat_class: MagicMock) -> None:
        """测试空证据时返回特定回答"""
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        generator = EvidenceConstrainedGenerator()
        result = generator.generate("查询", [])

        assert result.confidence_score == 0.0
        assert result.cited_chunks == []

    @patch("src.generation.evidence_constrained.ChatOpenAI")
    def test_generation_exception_handling(self, mock_chat_class: MagicMock) -> None:
        """测试异常处理"""
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API连接失败")

        generator = EvidenceConstrainedGenerator()
        evidence = [make_chunk_result("chunk_0001")]
        result = generator.generate("查询", evidence)

        assert result.confidence_score == 0.0
        assert "API连接失败" in result.answer or "出错" in result.answer


class TestConfidenceGate:
    """ConfidenceGate测试类"""

    def test_confidence_gate_low_confidence(self) -> None:
        """测试低置信度触发澄清"""
        gate = ConfidenceGate(threshold=0.4, high_threshold=0.7)

        gen_result = GenerationResult(
            answer="不确定的回答",
            cited_chunks=[],
            confidence_score=0.1,
        )
        retrieval_results = [make_chunk_result("chunk_001", score=0.1)]

        gate_result = gate.evaluate("查询", gen_result, retrieval_results)

        assert gate_result.should_clarify is True
        assert gate_result.clarification is not None
        assert gate_result.confidence < 0.4

    def test_confidence_gate_high_confidence(self) -> None:
        """测试高置信度产生干净答案"""
        gate = ConfidenceGate(threshold=0.4, high_threshold=0.7)

        gen_result = GenerationResult(
            answer="经济舱旅客可以免费托运20公斤行李。",
            cited_chunks=["chunk_001", "chunk_002"],
            confidence_score=0.95,
        )
        retrieval_results = [
            make_chunk_result("chunk_001", score=0.95),
            make_chunk_result("chunk_002", score=0.90),
            make_chunk_result("chunk_003", score=0.85),
        ]

        gate_result = gate.evaluate("行李规定", gen_result, retrieval_results)

        assert gate_result.should_clarify is False
        assert gate_result.warning is None
        assert gate_result.confidence >= 0.7

    def test_confidence_gate_medium_confidence_warning(self) -> None:
        """测试中等置信度产生警告"""
        gate = ConfidenceGate(threshold=0.4, high_threshold=0.7)

        gen_result = GenerationResult(
            answer="可能的回答",
            cited_chunks=["chunk_001"],
            confidence_score=0.6,
        )
        retrieval_results = [
            make_chunk_result("chunk_001", score=0.6),
            make_chunk_result("chunk_002", score=0.5),
        ]

        gate_result = gate.evaluate("查询", gen_result, retrieval_results)

        assert gate_result.should_clarify is False

    def test_confidence_calculation(self) -> None:
        """测试置信度计算公式"""
        gate = ConfidenceGate(threshold=0.4, high_threshold=0.7)

        gen_result = GenerationResult(
            answer="回答",
            cited_chunks=["chunk_001"],
            confidence_score=1.0,
        )
        retrieval_results = [
            make_chunk_result("chunk_001", score=1.0),
        ]

        gate_result = gate.evaluate("查询", gen_result, retrieval_results)

        # evidence_consistency=1.0, evidence_coverage=1.0, retrieval_score=1.0
        # final = 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert abs(gate_result.confidence - 1.0) < 0.01


class TestPipelineEndToEnd:
    """GraphRAGPipeline端到端集成测试"""

    @patch("src.pipeline.rag_pipeline.SemanticSplitter")
    @patch("src.pipeline.rag_pipeline.FAISSIndex")
    @patch("src.pipeline.rag_pipeline.BM25Index")
    @patch("src.pipeline.rag_pipeline.GraphBuilder")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.EvidenceConstrainedGenerator")
    def test_pipeline_query_end_to_end(
        self,
        mock_gen_class,
        mock_reranker_class,
        mock_graph_class,
        mock_bm25_class,
        mock_faiss_class,
        mock_splitter_class,
    ) -> None:
        """测试端到端查询流程返回正确结构"""
        from src.pipeline.rag_pipeline import GraphRAGPipeline, PipelineResult

        # 设置mock
        mock_faiss = MagicMock()
        mock_faiss_class.return_value = mock_faiss
        mock_faiss.index = MagicMock()
        mock_faiss.chunks = []

        mock_bm25 = MagicMock()
        mock_bm25_class.return_value = mock_bm25

        mock_graph = MagicMock()
        mock_graph_class.return_value = mock_graph
        mock_graph.graph.nodes = {}

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.rerank.return_value = [
            make_chunk_result("chunk_001", score=0.9),
            make_chunk_result("chunk_002", score=0.8),
        ]

        mock_gen = MagicMock()
        mock_gen_class.return_value = mock_gen
        mock_gen.generate.return_value = GenerationResult(
            answer="经济舱免费托运20公斤。",
            cited_chunks=["chunk_001"],
            confidence_score=0.85,
        )

        with patch("src.pipeline.rag_pipeline.HybridRetriever") as mock_hybrid_class:
            mock_hybrid = MagicMock()
            mock_hybrid_class.return_value = mock_hybrid
            mock_hybrid.retrieve.return_value = [
                make_chunk_result("chunk_001", score=0.9),
                make_chunk_result("chunk_002", score=0.8),
            ]

            pipeline = GraphRAGPipeline(config_path="config/settings.yaml")
            pipeline.indices_loaded = True

            result = pipeline.query("行李规定是什么？")

        assert isinstance(result, PipelineResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.latency_ms, float)
        assert isinstance(result.evidence_chunks, list)
        assert isinstance(result.graph_paths, list)

    @patch("src.pipeline.rag_pipeline.FAISSIndex")
    @patch("src.pipeline.rag_pipeline.BM25Index")
    @patch("src.pipeline.rag_pipeline.GraphBuilder")
    @patch("src.pipeline.rag_pipeline.CrossEncoderReranker")
    @patch("src.pipeline.rag_pipeline.EvidenceConstrainedGenerator")
    def test_pipeline_build_indices(
        self,
        mock_gen_class,
        mock_reranker_class,
        mock_graph_class,
        mock_bm25_class,
        mock_faiss_class,
    ) -> None:
        """测试构建索引功能"""
        from src.pipeline.rag_pipeline import GraphRAGPipeline

        mock_faiss = MagicMock()
        mock_faiss_class.return_value = mock_faiss
        mock_faiss.chunks = [MagicMock(), MagicMock()]

        mock_bm25 = MagicMock()
        mock_bm25_class.return_value = mock_bm25

        mock_graph = MagicMock()
        mock_graph_class.return_value = mock_graph
        mock_graph.graph.nodes.__len__ = MagicMock(return_value=5)

        mock_reranker_class.return_value = MagicMock()
        mock_gen_class.return_value = MagicMock()

        with patch("src.pipeline.rag_pipeline.SemanticSplitter") as mock_splitter_class:
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter

            from src.chunking.semantic_splitter import Chunk
            mock_chunks = [
                Chunk("chunk_0000", "文本一", {}),
                Chunk("chunk_0001", "文本二", {}),
            ]
            mock_splitter.split.return_value = mock_chunks

            # 先创建pipeline（会读取真实config文件），然后mock open用于build_indices中的数据读取
            pipeline = GraphRAGPipeline(config_path="config/settings.yaml")

            with patch("builtins.open", mock_open(read_data="# 测试文档\n内容")):
                with patch("os.makedirs"):
                    result = pipeline.build_indices("data/raw/test.md")

        assert "chunk_count" in result
        assert "node_count" in result
