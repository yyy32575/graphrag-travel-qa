# GraphRAG 出行知识问答系统

## 项目简介

本项目是一个基于知识图谱增强检索（GraphRAG）的出行知识问答系统，支持航空行李规定、铁路票务、签证要求、旅行保险等多维度出行规则的智能问答。

系统采用混合检索（密集+稀疏）+ 知识图谱扩展 + 交叉编码器重排序 + 证据约束生成的完整流水线，实现高质量的多跳推理问答。

## 架构图

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   GraphRAG 流水线                    │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  密集检索     │    │  稀疏检索    │               │
│  │ (FAISS+BGE)  │    │ (BM25+jieba) │               │
│  └──────┬───────┘    └──────┬───────┘               │
│         │                  │                        │
│         └────────┬─────────┘                        │
│                  ▼                                  │
│         ┌────────────────┐                          │
│         │   RRF混合融合   │                          │
│         └────────┬───────┘                          │
│                  ▼                                  │
│         ┌────────────────┐   ┌─────────────────┐    │
│         │ 交叉编码器重排序 │  │  知识图谱扩展    │    │
│         │ (BGE Reranker) │◄──│  (BFS多跳推理)   │    │
│         └────────┬───────┘   └─────────────────┘    │
│                  ▼                                  │
│         ┌────────────────┐                          │
│         │  证据约束生成   │                          │
│         │   (GPT-3.5)    │                          │
│         └────────┬───────┘                          │
│                  ▼                                  │
│         ┌────────────────┐                          │
│         │    置信度门控   │                          │
│         └────────┬───────┘                          │
└──────────────────┼──────────────────────────────────┘
                   ▼
              最终答案 + 置信度 + 证据链
```

## 快速开始

### 环境要求

- Python 3.11+
- 8GB+ 内存（用于嵌入模型）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 文件，填入 OpenAI API Key
```

### 构建索引

```bash
python scripts/build_index.py \
  --data-path data/raw/sample_travel_rules.md \
  --index-dir data/processed
```

### 启动 API 服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 部署

```bash
docker-compose up -d
```

## API 文档

服务启动后访问 http://localhost:8000/docs 查看交互式API文档。

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/query | 执行问答查询 |
| POST | /api/v1/index/build | 构建索引 |
| GET | /api/v1/health | 健康检查 |
| GET | /api/v1/stats | 统计信息 |

### 查询示例

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "带婴儿乘坐经济舱可以免费托运多少行李？",
    "top_k": 5,
    "enable_graph_expansion": true,
    "max_hops": 2
  }'
```

响应示例：

```json
{
  "answer": "带婴儿乘坐经济舱时，成人旅客可免费托运20公斤行李（规则A001），婴儿额外享有10公斤免费托运行李及一件折叠式婴儿推车（规则A008）。",
  "confidence": 0.85,
  "evidences": [...],
  "graph_paths": [...],
  "latency_ms": 1250.5
}
```

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试模块
python -m pytest tests/test_chunking.py -v
python -m pytest tests/test_retrieval.py -v
python -m pytest tests/test_graph.py -v
python -m pytest tests/test_pipeline.py -v
```

## 实验复现

### 分块参数网格搜索

```bash
python experiments/grid_search_chunking.py
# 结果保存到 experiments/results/grid_search_results.csv
```

### 检索方法消融实验

```bash
python experiments/retrieval_ablation.py
# 结果保存到 experiments/results/retrieval_ablation.csv
```

### 分桶评估

```bash
python experiments/eval_bucketed.py
# 结果保存到 experiments/results/eval_bucketed.csv
```

### 错误分析

```bash
python experiments/error_analysis.py
# 结果保存到 experiments/results/error_analysis.csv
```

## 项目结构

```
graphrag-travel-qa/
├── config/
│   └── settings.yaml          # 系统配置文件
├── data/
│   ├── raw/
│   │   └── sample_travel_rules.md  # 出行规则文档
│   ├── eval/
│   │   ├── complex_queries.jsonl   # 多跳推理评测集
│   │   └── general_queries.jsonl  # 通用评测集
│   └── processed/             # 索引文件（自动生成）
├── src/
│   ├── chunking/
│   │   ├── semantic_splitter.py   # 语义分块器
│   │   └── overlap_window.py     # 滑动窗口分块器
│   ├── indexing/
│   │   ├── faiss_index.py        # FAISS向量索引
│   │   ├── bm25_index.py         # BM25稀疏索引
│   │   └── graph_builder.py      # 知识图谱构建
│   ├── retrieval/
│   │   ├── dense_retriever.py    # 密集检索器
│   │   ├── sparse_retriever.py   # 稀疏检索器
│   │   ├── hybrid_retriever.py   # 混合检索（RRF）
│   │   ├── cross_encoder_reranker.py  # 交叉编码器重排
│   │   └── graph_expander.py     # 图谱扩展器
│   ├── generation/
│   │   ├── evidence_constrained.py  # 证据约束生成
│   │   └── confidence_gate.py       # 置信度门控
│   ├── pipeline/
│   │   └── rag_pipeline.py      # 端到端流水线
│   └── api/
│       ├── main.py             # FastAPI 应用入口
│       ├── routes.py           # API 路由定义
│       └── schemas.py          # Pydantic 数据模型
├── experiments/
│   ├── grid_search_chunking.py  # 分块参数搜索
│   ├── retrieval_ablation.py   # 检索消融实验
│   ├── eval_bucketed.py        # 分桶评估
│   └── error_analysis.py       # 错误分析
├── scripts/
│   ├── build_index.py          # 构建索引脚本
│   └── run_eval.py            # 运行评估脚本
├── tests/
│   ├── test_chunking.py        # 分块器测试
│   ├── test_retrieval.py       # 检索器测试
│   ├── test_graph.py           # 图谱测试
│   └── test_pipeline.py        # 流水线集成测试
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── requirements.txt
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 嵌入模型 | BAAI/bge-base-zh-v1.5 | 中文语义向量化 |
| 向量数据库 | FAISS (IndexFlatIP) | 高效余弦相似度检索 |
| 稀疏检索 | BM25Okapi + jieba | 中文关键词检索 |
| 知识图谱 | NetworkX (DiGraph) | 规则关系图谱 |
| 重排序 | BAAI/bge-reranker-base | 交叉编码器精排 |
| 生成模型 | GPT-3.5-turbo | 证据约束文本生成 |
| 推理框架 | LangChain | LLM调用封装 |
| API框架 | FastAPI + Uvicorn | 异步Web服务 |
| 数据验证 | Pydantic v2 | 请求响应模型 |

## 实验结果摘要

### 检索消融实验

| 配置 | 命中率 | 平均延迟(ms) |
|------|--------|-------------|
| 仅稀疏检索 (BM25) | ~0.65 | ~50 |
| 混合检索 (RRF) | ~0.78 | ~120 |
| 混合+重排序 | ~0.83 | ~350 |
| 混合+重排+图扩展 | ~0.87 | ~500 |

### 分桶评估结果

| 难度 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| 简单问题 | 0.82 | 0.79 | 0.80 |
| 多跳推理 | 0.71 | 0.68 | 0.69 |

> 注：以上为估算数据，实际结果依赖模型版本和数据质量。

## 许可证

MIT License
