"""GraphRAG流水线模块 - 整合所有组件的端到端问答流水线"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from src.chunking.semantic_splitter import SemanticSplitter
from src.indexing.faiss_index import FAISSIndex, ChunkResult
from src.indexing.bm25_index import BM25Index
from src.indexing.graph_builder import GraphBuilder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.graph_expander import GraphExpander, GraphPath
from src.generation.evidence_constrained import EvidenceConstrainedGenerator
from src.generation.confidence_gate import ConfidenceGate


@dataclass
class PipelineResult:
    """流水线结果数据类"""
    answer: str
    evidence_chunks: List[ChunkResult]
    graph_paths: List[Any]  # List[GraphPath]
    confidence: float
    latency_ms: float
    stage_details: Dict[str, Any]
    clarification: Optional[str]


class GraphRAGPipeline:
    """GraphRAG问答流水线
    
    整合语义分块、密集/稀疏检索、知识图谱扩展、
    交叉编码器重排序和证据约束生成的完整流水线。
    """

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """初始化流水线（不加载索引）
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        emb_cfg = self.config.get("embedding", {})
        ret_cfg = self.config.get("retrieval", {})
        gen_cfg = self.config.get("generation", {})

        # 初始化索引组件
        self.faiss_index = FAISSIndex(
            model_name=emb_cfg.get("model_name", "BAAI/bge-base-zh-v1.5"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=emb_cfg.get("batch_size", 32),
        )
        self.bm25_index = BM25Index()
        self.graph_builder = GraphBuilder()

        # 初始化检索组件
        self.dense_retriever = DenseRetriever(self.faiss_index)
        self.sparse_retriever = SparseRetriever(self.bm25_index)

        hybrid_cfg = ret_cfg.get("hybrid", {})
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            rrf_k=hybrid_cfg.get("rrf_k", 60),
            dense_top_k=ret_cfg.get("dense", {}).get("top_k", 20),
            sparse_top_k=ret_cfg.get("sparse", {}).get("top_k", 20),
        )

        reranker_cfg = ret_cfg.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-base"),
            batch_size=reranker_cfg.get("batch_size", 16),
        )

        graph_cfg = ret_cfg.get("graph_expansion", {})
        self.graph_expander = GraphExpander(
            graph_builder=self.graph_builder,
            max_hops=graph_cfg.get("max_hops", 2),
            decay_factor=graph_cfg.get("decay_factor", 0.7),
            min_path_score=graph_cfg.get("min_path_score", 0.1),
            max_expanded_chunks=graph_cfg.get("max_expanded_chunks", 5),
        )

        # 初始化生成组件
        self.generator = EvidenceConstrainedGenerator(
            model_name=gen_cfg.get("model_name", "gpt-3.5-turbo"),
            temperature=gen_cfg.get("temperature", 0.1),
            max_tokens=gen_cfg.get("max_tokens", 1024),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

        self.confidence_gate = ConfidenceGate(
            threshold=gen_cfg.get("confidence_threshold", 0.4),
        )

        self.indices_loaded = False

    def load_indices(self, index_dir: str = "data/processed") -> None:
        """从磁盘加载索引
        
        Args:
            index_dir: 索引文件目录
        """
        faiss_path = os.path.join(index_dir, "faiss_index")
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")

        self.faiss_index.load(faiss_path)
        self.bm25_index.load(bm25_path)
        self.indices_loaded = True

    def build_indices(
        self,
        data_path: str,
        index_dir: str = "data/processed",
    ) -> Dict[str, int]:
        """从原始数据构建并保存所有索引
        
        Args:
            data_path: 原始数据文件路径（Markdown格式）
            index_dir: 索引保存目录
            
        Returns:
            包含chunk_count和node_count的字典
        """
        # 读取数据
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 语义分块
        chunk_cfg = self.config.get("chunking", {})
        semantic_cfg = chunk_cfg.get("semantic", {})
        emb_cfg = self.config.get("embedding", {})

        splitter = SemanticSplitter(
            model_name=emb_cfg.get("model_name", "BAAI/bge-base-zh-v1.5"),
            similarity_threshold=semantic_cfg.get("similarity_threshold", 0.5),
            min_chunk_size=semantic_cfg.get("min_chunk_size", 100),
            max_chunk_size=semantic_cfg.get("max_chunk_size", 1024),
            device=emb_cfg.get("device", "cpu"),
        )
        chunks = splitter.split(text)

        if not chunks:
            return {"chunk_count": 0, "node_count": 0}

        # 构建索引
        self.faiss_index.build(chunks)
        self.bm25_index.build(chunks)
        self.graph_builder.build_from_chunks(chunks)

        # 保存索引
        os.makedirs(index_dir, exist_ok=True)
        faiss_path = os.path.join(index_dir, "faiss_index")
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")

        self.faiss_index.save(faiss_path)
        self.bm25_index.save(bm25_path)

        self.indices_loaded = True

        return {
            "chunk_count": len(chunks),
            "node_count": len(self.graph_builder.graph.nodes),
        }

    def query(
        self,
        question: str,
        top_k: int = 5,
        enable_graph_expansion: bool = True,
        max_hops: int = 2,
    ) -> PipelineResult:
        """执行端到端问答
        
        Args:
            question: 用户问题
            top_k: 最终返回的证据块数量
            enable_graph_expansion: 是否启用图谱扩展
            max_hops: 图谱扩展最大跳数
            
        Returns:
            PipelineResult对象
        """
        start_time = time.time()
        stage_details: Dict[str, Any] = {}

        # 1. 混合检索
        reranker_cfg = self.config.get("retrieval", {}).get("reranker", {})
        rerank_top_k = reranker_cfg.get("top_k", 10)

        hybrid_results = self.hybrid_retriever.retrieve(question, top_k=rerank_top_k)
        stage_details["hybrid_retrieval_count"] = len(hybrid_results)

        # 2. 重排序
        reranked_results = self.reranker.rerank(question, hybrid_results, top_k=rerank_top_k)
        stage_details["reranked_count"] = len(reranked_results)

        # 3. 图谱扩展（可选）
        graph_paths: List[GraphPath] = []
        if enable_graph_expansion and self.graph_builder.graph.nodes:
            chunk_ids = [r.chunk_id for r in reranked_results[:top_k]]
            # 临时调整max_hops
            self.graph_expander.max_hops = max_hops
            expanded_results, graph_paths = self.graph_expander.expand(
                chunk_ids, reranked_results
            )
            evidence_chunks = expanded_results[:top_k]
            stage_details["graph_expanded_count"] = len(graph_paths)
        else:
            evidence_chunks = reranked_results[:top_k]
            stage_details["graph_expanded_count"] = 0

        # 4. 证据约束生成
        generation_result = self.generator.generate(question, evidence_chunks)
        stage_details["cited_chunks"] = generation_result.cited_chunks
        stage_details["generation_confidence"] = generation_result.confidence_score

        # 5. 置信度门控
        gate_result = self.confidence_gate.evaluate(
            question, generation_result, evidence_chunks
        )
        stage_details["final_confidence"] = gate_result.confidence

        latency_ms = (time.time() - start_time) * 1000

        return PipelineResult(
            answer=gate_result.answer,
            evidence_chunks=evidence_chunks,
            graph_paths=graph_paths,
            confidence=gate_result.confidence,
            latency_ms=latency_ms,
            stage_details=stage_details,
            clarification=gate_result.clarification,
        )
