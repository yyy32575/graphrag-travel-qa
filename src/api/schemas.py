"""API数据模型定义"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """问答请求模型"""
    question: str
    top_k: int = 5
    enable_graph_expansion: bool = True
    max_hops: int = 2


class Evidence(BaseModel):
    """证据块模型"""
    chunk_id: str
    text: str
    score: float
    source: str


class GraphPath(BaseModel):
    """图谱路径模型"""
    nodes: List[str]
    relations: List[str]
    score: float


class QueryResponse(BaseModel):
    """问答响应模型"""
    answer: str
    confidence: float
    evidences: List[Evidence]
    graph_paths: List[GraphPath]
    latency_ms: float
    clarification: Optional[str] = None


class IndexBuildRequest(BaseModel):
    """索引构建请求模型"""
    data_path: Optional[str] = None
    force_rebuild: bool = False


class IndexBuildResponse(BaseModel):
    """索引构建响应模型"""
    success: bool
    message: str
    chunk_count: int
    node_count: int


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    version: str
    indices_loaded: bool


class StatsResponse(BaseModel):
    """统计信息响应模型"""
    chunk_count: int
    node_count: int
    edge_count: int
    index_loaded: bool
