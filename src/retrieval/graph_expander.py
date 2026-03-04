"""图谱扩展模块 - 基于BFS的知识图谱关联扩展"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.indexing.faiss_index import ChunkResult
from src.indexing.graph_builder import GraphBuilder, NodeType


@dataclass
class GraphPath:
    """图遍历路径数据类"""
    nodes: List[str]
    relations: List[str]
    score: float


class GraphExpander:
    """知识图谱扩展器
    
    从初始检索结果出发，通过BFS遍历知识图谱，
    找到相关联的文本块，扩展上下文信息。
    分数随跳数以decay_factor衰减。
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        max_hops: int = 2,
        decay_factor: float = 0.7,
        min_path_score: float = 0.1,
        max_expanded_chunks: int = 5,
    ) -> None:
        """初始化图谱扩展器
        
        Args:
            graph_builder: 知识图谱构建器
            max_hops: 最大跳数
            decay_factor: 每跳分数衰减因子
            min_path_score: 最小路径分数阈值
            max_expanded_chunks: 最大扩展块数量
        """
        self.graph_builder = graph_builder
        self.max_hops = max_hops
        self.decay_factor = decay_factor
        self.min_path_score = min_path_score
        self.max_expanded_chunks = max_expanded_chunks

    def _get_chunk_nodes(self, chunk_id: str) -> List[str]:
        """获取与指定chunk_id关联的图节点
        
        Args:
            chunk_id: 文本块ID
            
        Returns:
            节点ID列表
        """
        nodes = []
        for node_id, data in self.graph_builder.graph.nodes(data=True):
            if data.get("source_chunk_id") == chunk_id:
                nodes.append(node_id)
        return nodes

    def expand(
        self,
        chunk_ids: List[str],
        initial_results: List[ChunkResult],
    ) -> Tuple[List[ChunkResult], List[GraphPath]]:
        """从初始块出发通过图谱扩展关联内容
        
        Args:
            chunk_ids: 初始文本块ID列表（用于图谱扩展的起点）
            initial_results: 初始检索结果
            
        Returns:
            (combined_results, graph_paths) 元组：
            - combined_results: 原始结果 + 扩展结果的合并列表
            - graph_paths: 遍历路径列表
        """
        if not self.graph_builder.graph.nodes:
            return initial_results, []

        # 构建已有chunk_id到score的映射
        existing_chunk_scores: Dict[str, float] = {
            r.chunk_id: r.score for r in initial_results
        }
        existing_chunk_map: Dict[str, ChunkResult] = {
            r.chunk_id: r for r in initial_results
        }

        # 找到起始节点
        start_nodes: List[Tuple[str, float]] = []
        for chunk_id in chunk_ids:
            nodes = self._get_chunk_nodes(chunk_id)
            initial_score = existing_chunk_scores.get(chunk_id, 0.5)
            for node_id in nodes:
                start_nodes.append((node_id, initial_score))

        if not start_nodes:
            return initial_results, []

        # BFS遍历
        visited: Dict[str, float] = {}
        graph_paths: List[GraphPath] = []
        expanded_chunks: Dict[str, float] = {}

        queue: deque = deque()
        for node_id, score in start_nodes:
            queue.append((node_id, score, [node_id], []))
            visited[node_id] = score

        while queue:
            current_node, current_score, path_nodes, path_relations = queue.popleft()

            if len(path_nodes) > self.max_hops + 1:
                continue

            # 遍历邻居
            for neighbor in self.graph_builder.graph.successors(current_node):
                edge_data = self.graph_builder.graph.edges[current_node, neighbor]
                edge_weight = edge_data.get("weight", 1.0)
                relation_type = edge_data.get("relation_type", "related")
                
                new_score = current_score * self.decay_factor * edge_weight

                if new_score < self.min_path_score:
                    continue

                new_path_nodes = path_nodes + [neighbor]
                new_path_relations = path_relations + [relation_type]

                if len(new_path_nodes) > 1:
                    graph_paths.append(GraphPath(
                        nodes=new_path_nodes,
                        relations=new_path_relations,
                        score=new_score,
                    ))

                if neighbor not in visited or visited[neighbor] < new_score:
                    visited[neighbor] = new_score

                    # 获取该节点关联的chunk_id
                    node_data = self.graph_builder.graph.nodes[neighbor]
                    neighbor_chunk_id = node_data.get("source_chunk_id")
                    
                    if (neighbor_chunk_id and 
                        neighbor_chunk_id not in existing_chunk_scores and
                        neighbor_chunk_id not in expanded_chunks):
                        expanded_chunks[neighbor_chunk_id] = new_score

                    if len(new_path_nodes) <= self.max_hops:
                        queue.append((neighbor, new_score, new_path_nodes, new_path_relations))

        # 按分数排序扩展块，取前max_expanded_chunks个
        sorted_expanded = sorted(
            expanded_chunks.items(), key=lambda x: x[1], reverse=True
        )[:self.max_expanded_chunks]

        # 从图中获取扩展块的文本
        chunk_id_to_text: Dict[str, str] = {}
        chunk_id_to_metadata: Dict[str, Dict] = {}
        for node_id, data in self.graph_builder.graph.nodes(data=True):
            src = data.get("source_chunk_id")
            if src and src not in chunk_id_to_text:
                chunk_id_to_text[src] = data.get("text", "")
                chunk_id_to_metadata[src] = data.get("metadata", {})

        expanded_results: List[ChunkResult] = []
        for chunk_id, score in sorted_expanded:
            expanded_results.append(ChunkResult(
                chunk_id=chunk_id,
                text=chunk_id_to_text.get(chunk_id, ""),
                score=score,
                metadata={
                    **chunk_id_to_metadata.get(chunk_id, {}),
                    "expanded_via_graph": True,
                },
            ))

        # 合并原始结果和扩展结果
        combined_results = list(initial_results) + expanded_results
        
        # 按分数排序
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results, graph_paths
