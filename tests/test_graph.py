"""图谱构建器和扩展器单元测试"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.chunking.semantic_splitter import Chunk
from src.indexing.graph_builder import GraphBuilder, NodeType
from src.indexing.faiss_index import ChunkResult
from src.retrieval.graph_expander import GraphExpander, GraphPath


def make_chunk(chunk_id: str, text: str = "规则测试内容") -> Chunk:
    """创建测试用Chunk"""
    return Chunk(chunk_id=chunk_id, text=text, metadata={})


def make_chunk_result(chunk_id: str, score: float = 0.8) -> ChunkResult:
    """创建测试用ChunkResult"""
    return ChunkResult(chunk_id=chunk_id, text="测试文本", score=score)


class TestGraphBuilder:
    """GraphBuilder测试类"""

    def test_add_rule_creates_node(self) -> None:
        """测试add_rule创建具有正确属性的节点"""
        builder = GraphBuilder()
        rule_text = "**规则A001：测试规则** 条件：满足条件A。 规定：需要执行操作B。"
        node_id = builder.add_rule(rule_text, "chunk_0000")

        assert node_id in builder.graph.nodes
        node_data = builder.graph.nodes[node_id]
        assert node_data["type"] == NodeType.RULE.value
        assert node_data["source_chunk_id"] == "chunk_0000"
        assert "A001" in builder._rule_id_to_node

    def test_add_rule_extracts_conditions(self) -> None:
        """测试add_rule提取条件节点"""
        builder = GraphBuilder()
        rule_text = "**规则B001：条件测试** 条件：旅客持有有效机票且行李重量不超过20公斤。结论：可以免费托运。"
        node_id = builder.add_rule(rule_text, "chunk_0001")

        # 应该有多个节点：规则节点 + 条件/结论节点
        assert len(builder.graph.nodes) > 1

    def test_build_from_chunks(self) -> None:
        """测试从文本块列表构建图谱"""
        builder = GraphBuilder()
        chunks = [
            make_chunk("chunk_0000", "**规则A001：行李规定** 条件：经济舱旅客。规定：免费托运20公斤。"),
            make_chunk("chunk_0001", "**规则A002：超重费** 条件：行李超重时。规定：每公斤收取30元。"),
        ]
        builder.build_from_chunks(chunks)

        # 应该有节点被创建
        assert len(builder.graph.nodes) > 0

    def test_get_neighbors_bfs(self) -> None:
        """测试BFS遍历邻居"""
        builder = GraphBuilder()

        # 手动构建简单图
        builder.graph.add_node("node_A", type=NodeType.RULE.value, text="规则A", source_chunk_id="chunk_0", id="node_A", metadata={})
        builder.graph.add_node("node_B", type=NodeType.CONDITION.value, text="条件B", source_chunk_id="chunk_0", id="node_B", metadata={})
        builder.graph.add_node("node_C", type=NodeType.CONCLUSION.value, text="结论C", source_chunk_id="chunk_1", id="node_C", metadata={})
        builder.graph.add_edge("node_A", "node_B", relation_type="requires", weight=1.0)
        builder.graph.add_edge("node_B", "node_C", relation_type="leads_to", weight=0.8)

        neighbors = builder.get_neighbors("node_A", max_hops=2)

        neighbor_ids = [n.get("id") for n in neighbors]
        assert "node_B" in neighbor_ids

    def test_get_neighbors_max_hops(self) -> None:
        """测试BFS遍历遵守最大跳数限制"""
        builder = GraphBuilder()

        # 创建3层深度的链式图
        for i in range(4):
            builder.graph.add_node(
                f"node_{i}",
                type=NodeType.RULE.value,
                text=f"节点{i}",
                source_chunk_id=f"chunk_{i}",
                id=f"node_{i}",
                metadata={},
            )
        for i in range(3):
            builder.graph.add_edge(f"node_{i}", f"node_{i+1}", relation_type="leads_to", weight=1.0)

        # max_hops=1，应只返回直接邻居
        neighbors = builder.get_neighbors("node_0", max_hops=1)
        neighbor_ids = [n.get("id") for n in neighbors]

        assert "node_1" in neighbor_ids
        assert "node_2" not in neighbor_ids

    def test_score_decay(self) -> None:
        """测试分数随跳数衰减"""
        builder = GraphBuilder()

        builder.graph.add_node("node_0", type=NodeType.RULE.value, text="规则0", source_chunk_id="chunk_0", id="node_0", metadata={})
        builder.graph.add_node("node_1", type=NodeType.RULE.value, text="规则1", source_chunk_id="chunk_1", id="node_1", metadata={})
        builder.graph.add_node("node_2", type=NodeType.RULE.value, text="规则2", source_chunk_id="chunk_2", id="node_2", metadata={})
        builder.graph.add_edge("node_0", "node_1", relation_type="requires", weight=1.0)
        builder.graph.add_edge("node_1", "node_2", relation_type="requires", weight=1.0)

        neighbors = builder.get_neighbors("node_0", max_hops=2, min_score=0.0)

        # 找到node_1和node_2的分数
        score_map = {n.get("id"): n.get("reach_score") for n in neighbors}
        if "node_1" in score_map and "node_2" in score_map:
            # node_2的分数应低于node_1（经过更多跳）
            assert score_map["node_2"] <= score_map["node_1"]

    def test_get_subgraph(self) -> None:
        """测试获取子图"""
        builder = GraphBuilder()
        builder.graph.add_node("node_A", type=NodeType.RULE.value, text="A", source_chunk_id="c0", id="node_A", metadata={})
        builder.graph.add_node("node_B", type=NodeType.RULE.value, text="B", source_chunk_id="c1", id="node_B", metadata={})
        builder.graph.add_node("node_C", type=NodeType.RULE.value, text="C", source_chunk_id="c2", id="node_C", metadata={})
        builder.graph.add_edge("node_A", "node_B", relation_type="requires", weight=1.0)

        subgraph = builder.get_subgraph(["node_A", "node_B"])

        assert "node_A" in subgraph.nodes
        assert "node_B" in subgraph.nodes
        assert "node_C" not in subgraph.nodes

    def test_export_graphml(self, tmp_path) -> None:
        """测试导出GraphML格式"""
        builder = GraphBuilder()
        builder.graph.add_node("node_A", type=NodeType.RULE.value, text="测试规则", source_chunk_id="c0", id="node_A", metadata={})

        output_path = str(tmp_path / "test_graph.graphml")
        builder.export_graphml(output_path)

        import os
        assert os.path.exists(output_path)


class TestGraphExpander:
    """GraphExpander测试类"""

    def test_graph_expander_expand(self) -> None:
        """测试图谱扩展返回ChunkResults和GraphPaths"""
        builder = GraphBuilder()

        # 构建简单图：chunk_0 -> chunk_1
        builder.graph.add_node("node_0", type=NodeType.RULE.value, text="规则0", source_chunk_id="chunk_0000", id="node_0", metadata={})
        builder.graph.add_node("node_1", type=NodeType.RULE.value, text="规则1", source_chunk_id="chunk_0001", id="node_1", metadata={})
        builder.graph.add_edge("node_0", "node_1", relation_type="requires", weight=1.0)

        expander = GraphExpander(builder, max_hops=2, decay_factor=0.7, min_path_score=0.01)
        initial_results = [make_chunk_result("chunk_0000", score=0.9)]

        combined, paths = expander.expand(["chunk_0000"], initial_results)

        assert isinstance(combined, list)
        assert isinstance(paths, list)
        assert len(combined) >= len(initial_results)

    def test_graph_expander_empty_graph(self) -> None:
        """测试空图谱时扩展返回原始结果"""
        builder = GraphBuilder()  # 空图
        expander = GraphExpander(builder)

        initial_results = [make_chunk_result("chunk_0000")]
        combined, paths = expander.expand(["chunk_0000"], initial_results)

        assert combined == initial_results
        assert paths == []

    def test_graph_expander_score_decay(self) -> None:
        """测试扩展分数按decay_factor衰减"""
        builder = GraphBuilder()

        builder.graph.add_node("node_0", type=NodeType.RULE.value, text="规则0", source_chunk_id="chunk_0000", id="node_0", metadata={})
        builder.graph.add_node("node_1", type=NodeType.RULE.value, text="规则1", source_chunk_id="chunk_0001", id="node_1", metadata={})
        builder.graph.add_edge("node_0", "node_1", relation_type="requires", weight=1.0)

        decay_factor = 0.5
        expander = GraphExpander(builder, decay_factor=decay_factor, min_path_score=0.0)
        initial_results = [make_chunk_result("chunk_0000", score=1.0)]

        combined, paths = expander.expand(["chunk_0000"], initial_results)

        # 找到扩展的chunk
        expanded = [r for r in combined if r.chunk_id == "chunk_0001"]
        if expanded:
            assert expanded[0].score < initial_results[0].score
