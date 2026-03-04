"""API路由定义"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    Evidence,
    GraphPath,
    IndexBuildRequest,
    IndexBuildResponse,
    HealthResponse,
    StatsResponse,
)

if TYPE_CHECKING:
    from src.pipeline.rag_pipeline import GraphRAGPipeline

router = APIRouter()

# 全局流水线引用，由main.py设置
_pipeline: "GraphRAGPipeline | None" = None


def set_pipeline(pipeline: "GraphRAGPipeline") -> None:
    """设置全局流水线引用
    
    Args:
        pipeline: GraphRAGPipeline实例
    """
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> "GraphRAGPipeline":
    """获取流水线实例
    
    Returns:
        GraphRAGPipeline实例
        
    Raises:
        HTTPException: 流水线未初始化时
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="流水线未初始化")
    return _pipeline


@router.post("/query", response_model=QueryResponse, summary="执行问答查询")
async def query(request: QueryRequest) -> QueryResponse:
    """执行GraphRAG问答查询
    
    Args:
        request: 查询请求
        
    Returns:
        查询响应
    """
    pipeline = get_pipeline()

    if not pipeline.indices_loaded:
        raise HTTPException(status_code=503, detail="索引尚未加载，请先构建或加载索引")

    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            enable_graph_expansion=request.enable_graph_expansion,
            max_hops=request.max_hops,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询执行失败：{str(e)}")

    # 转换证据格式
    evidences = [
        Evidence(
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
            source=c.metadata.get("source", "unknown"),
        )
        for c in result.evidence_chunks
    ]

    # 转换图谱路径格式
    graph_paths = [
        GraphPath(
            nodes=p.nodes,
            relations=p.relations,
            score=p.score,
        )
        for p in result.graph_paths
    ]

    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        evidences=evidences,
        graph_paths=graph_paths,
        latency_ms=result.latency_ms,
        clarification=result.clarification,
    )


@router.post("/index/build", response_model=IndexBuildResponse, summary="构建索引")
async def build_index(request: IndexBuildRequest) -> IndexBuildResponse:
    """构建或重建所有索引
    
    Args:
        request: 索引构建请求
        
    Returns:
        构建结果
    """
    pipeline = get_pipeline()

    data_path = request.data_path or "data/raw/sample_travel_rules.md"

    try:
        stats = pipeline.build_indices(data_path=data_path)
        return IndexBuildResponse(
            success=True,
            message=f"索引构建成功",
            chunk_count=stats["chunk_count"],
            node_count=stats["node_count"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"数据文件未找到：{data_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引构建失败：{str(e)}")


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    """健康检查端点
    
    Returns:
        服务健康状态
    """
    if _pipeline is None:
        return HealthResponse(
            status="initializing",
            version="0.1.0",
            indices_loaded=False,
        )

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        indices_loaded=_pipeline.indices_loaded,
    )


@router.get("/stats", response_model=StatsResponse, summary="获取统计信息")
async def stats() -> StatsResponse:
    """获取系统统计信息
    
    Returns:
        统计信息
    """
    pipeline = get_pipeline()

    chunk_count = len(pipeline.faiss_index.chunks) if pipeline.faiss_index.index is not None else 0
    node_count = len(pipeline.graph_builder.graph.nodes)
    edge_count = len(pipeline.graph_builder.graph.edges)

    return StatsResponse(
        chunk_count=chunk_count,
        node_count=node_count,
        edge_count=edge_count,
        index_loaded=pipeline.indices_loaded,
    )
