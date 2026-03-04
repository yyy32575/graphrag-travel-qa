"""知识图谱构建模块 - 基于NetworkX的规则关系图"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示错误
import matplotlib.pyplot as plt
import networkx as nx

from src.chunking.semantic_splitter import Chunk


class NodeType(Enum):
    """图节点类型枚举"""
    RULE = "rule"
    CONDITION = "condition"
    CONCLUSION = "conclusion"


class GraphBuilder:
    """知识图谱构建器
    
    从文本块中提取实体和关系，构建有向知识图谱。
    使用正则表达式识别中文规则文本中的条件-结论关系。
    """

    # 条件模式关键词
    _CONDITION_PATTERNS = [
        r'条件[：:]\s*(.+?)(?=[。\n]|$)',
        r'如果(.+?)则',
        r'当(.+?)时',
        r'适用条件[：:]\s*(.+?)(?=[。\n]|$)',
        r'前提[：:]\s*(.+?)(?=[。\n]|$)',
    ]

    # 结论/要求模式关键词
    _CONCLUSION_PATTERNS = [
        r'则(.+?)(?=[。\n]|$)',
        r'需要(.+?)(?=[。\n]|$)',
        r'要求(.+?)(?=[。\n]|$)',
        r'规定[：:]\s*(.+?)(?=[。\n]|$)',
        r'须(.+?)(?=[。\n]|$)',
    ]

    # 关联规则模式
    _REFERENCE_PATTERNS = [
        r'参见规则([A-Z]\d+)',
        r'关联规则[：:].+?规则([A-Z]\d+)',
        r'详见([A-Z]\d+)',
    ]

    # 规则ID模式
    _RULE_ID_PATTERN = r'\*\*规则([A-Z]\d+)[：:]'

    def __init__(self) -> None:
        """初始化图谱构建器"""
        self.graph = nx.DiGraph()
        self._rule_id_to_node: Dict[str, str] = {}
        self._node_counter = 0

    def _new_node_id(self) -> str:
        """生成新节点ID"""
        node_id = f"node_{self._node_counter:04d}"
        self._node_counter += 1
        return node_id

    def add_rule(self, rule_text: str, source_chunk_id: str) -> str:
        """添加规则节点到图谱
        
        Args:
            rule_text: 规则文本
            source_chunk_id: 来源块ID
            
        Returns:
            新创建的节点ID
        """
        node_id = self._new_node_id()
        
        # 提取规则ID（如A001, R002等）
        rule_id_match = re.search(r'规则([A-Z]\d+)', rule_text)
        rule_id = rule_id_match.group(1) if rule_id_match else node_id

        self.graph.add_node(
            node_id,
            id=node_id,
            type=NodeType.RULE.value,
            text=rule_text[:200],  # 截断过长文本
            source_chunk_id=source_chunk_id,
            rule_id=rule_id,
            metadata={},
        )
        self._rule_id_to_node[rule_id] = node_id

        # 提取条件节点
        for pattern in self._CONDITION_PATTERNS:
            matches = re.findall(pattern, rule_text, re.DOTALL)
            for match in matches:
                match = match.strip()
                if len(match) > 5:  # 过滤太短的匹配
                    cond_node_id = self._new_node_id()
                    self.graph.add_node(
                        cond_node_id,
                        id=cond_node_id,
                        type=NodeType.CONDITION.value,
                        text=match[:200],
                        source_chunk_id=source_chunk_id,
                        metadata={},
                    )
                    self.graph.add_edge(
                        node_id, cond_node_id,
                        relation_type="requires",
                        weight=1.0,
                    )

        # 提取结论节点
        for pattern in self._CONCLUSION_PATTERNS:
            matches = re.findall(pattern, rule_text, re.DOTALL)
            for match in matches:
                match = match.strip()
                if len(match) > 5:
                    conc_node_id = self._new_node_id()
                    self.graph.add_node(
                        conc_node_id,
                        id=conc_node_id,
                        type=NodeType.CONCLUSION.value,
                        text=match[:200],
                        source_chunk_id=source_chunk_id,
                        metadata={},
                    )
                    self.graph.add_edge(
                        node_id, conc_node_id,
                        relation_type="leads_to",
                        weight=0.8,
                    )

        return node_id

    def _link_references(self) -> None:
        """链接跨规则引用关系"""
        for node_id, data in list(self.graph.nodes(data=True)):
            if data.get("type") != NodeType.RULE.value:
                continue
            text = data.get("text", "")
            for pattern in self._REFERENCE_PATTERNS:
                refs = re.findall(pattern, text)
                for ref in refs:
                    if ref in self._rule_id_to_node:
                        target_node = self._rule_id_to_node[ref]
                        if not self.graph.has_edge(node_id, target_node):
                            self.graph.add_edge(
                                node_id, target_node,
                                relation_type="requires",
                                weight=0.9,
                            )

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        """从文本块列表构建知识图谱
        
        Args:
            chunks: Chunk对象列表
        """
        for chunk in chunks:
            # 按规则标题分割块内容
            rule_sections = re.split(r'(?=\*\*规则[A-Z]\d+)', chunk.text)
            for section in rule_sections:
                section = section.strip()
                if section and len(section) > 20:
                    self.add_rule(section, chunk.chunk_id)

        # 建立跨块引用关系
        self._link_references()

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 2,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """BFS遍历获取节点的邻居
        
        Args:
            node_id: 起始节点ID
            max_hops: 最大跳数
            min_score: 最小权重分数
            
        Returns:
            邻居节点信息列表（包含节点属性和到达分数）
        """
        if node_id not in self.graph:
            return []

        visited: Dict[str, float] = {node_id: 1.0}
        queue = [(node_id, 0, 1.0)]
        result = []

        while queue:
            current_id, hops, current_score = queue.pop(0)
            if hops >= max_hops:
                continue

            for neighbor in self.graph.successors(current_id):
                edge_data = self.graph.edges[current_id, neighbor]
                edge_weight = edge_data.get("weight", 1.0)
                new_score = current_score * edge_weight

                if new_score < min_score:
                    continue

                if neighbor not in visited or visited[neighbor] < new_score:
                    visited[neighbor] = new_score
                    node_data = dict(self.graph.nodes[neighbor])
                    node_data["reach_score"] = new_score
                    node_data["hops"] = hops + 1
                    result.append(node_data)
                    queue.append((neighbor, hops + 1, new_score))

        return result

    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph:
        """获取指定节点的子图
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            包含指定节点的有向子图
        """
        # 只保留存在于图中的节点
        valid_ids = [n for n in node_ids if n in self.graph]
        return self.graph.subgraph(valid_ids).copy()

    def export_graphml(self, path: str) -> None:
        """导出图谱为GraphML格式
        
        Args:
            path: 导出文件路径
        """
        # 转换枚举类型为字符串，GraphML不支持枚举
        export_graph = self.graph.copy()
        for node_id, data in export_graph.nodes(data=True):
            if "metadata" in data:
                export_graph.nodes[node_id]["metadata"] = str(data["metadata"])
        nx.write_graphml(export_graph, path)

    def visualize(self, path: str) -> None:
        """可视化图谱并保存为图片
        
        Args:
            path: 图片保存路径
        """
        plt.figure(figsize=(16, 12))
        
        if len(self.graph.nodes) == 0:
            plt.text(0.5, 0.5, "图谱为空", ha='center', va='center', fontsize=16)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return

        # 按节点类型分组着色
        color_map = {
            NodeType.RULE.value: "#4CAF50",
            NodeType.CONDITION.value: "#2196F3",
            NodeType.CONCLUSION.value: "#FF9800",
        }
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", NodeType.RULE.value)
            node_colors.append(color_map.get(node_type, "#9E9E9E"))

        try:
            pos = nx.spring_layout(self.graph, seed=42, k=2)
        except Exception:
            pos = nx.random_layout(self.graph, seed=42)

        nx.draw_networkx(
            self.graph,
            pos=pos,
            node_color=node_colors,
            with_labels=False,
            node_size=300,
            arrows=True,
            arrowsize=10,
        )
        plt.title("知识图谱可视化", fontsize=14)
        plt.axis('off')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
