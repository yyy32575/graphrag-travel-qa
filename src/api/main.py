"""FastAPI应用入口"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.pipeline.rag_pipeline import GraphRAGPipeline
from src.api.routes import router, set_pipeline

# 全局流水线实例
pipeline: Optional[GraphRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理
    
    启动时初始化流水线并尝试加载索引，
    关闭时清理资源。
    """
    global pipeline

    config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")

    try:
        pipeline = GraphRAGPipeline(config_path=config_path)
        set_pipeline(pipeline)

        # 尝试加载已有索引
        index_dir = os.environ.get("INDEX_DIR", "data/processed")
        faiss_path = os.path.join(index_dir, "faiss_index.faiss")
        if os.path.exists(faiss_path):
            try:
                pipeline.load_indices(index_dir=index_dir)
                print(f"✓ 索引加载成功，共 {len(pipeline.faiss_index.chunks)} 个块")
            except Exception as e:
                print(f"⚠ 索引加载失败：{e}，请通过API构建索引")
        else:
            print("ℹ 未找到已有索引，请通过 POST /api/v1/index/build 构建索引")
    except Exception as e:
        print(f"✗ 流水线初始化失败：{e}")

    yield

    # 清理资源
    pipeline = None


def _load_cors_origins() -> list:
    """从配置文件加载CORS origins"""
    try:
        config_path = os.environ.get("CONFIG_PATH", "config/settings.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("api", {}).get("cors_origins", ["*"])
    except Exception:
        return ["*"]


# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG出行知识问答系统",
    version="0.1.0",
    description="基于知识图谱增强检索的出行规则问答系统",
    lifespan=lifespan,
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """ValueError异常处理器"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"内部服务器错误：{str(exc)}"},
    )
