"""交叉编码器重排序模块 - 使用BGE Reranker精排检索结果"""
from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from src.indexing.faiss_index import ChunkResult


class CrossEncoderReranker:
    """交叉编码器重排序器
    
    使用CrossEncoder模型对候选文本块进行精排，
    提高检索结果的相关性。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 16,
    ) -> None:
        """初始化重排序器
        
        Args:
            model_name: CrossEncoder模型名称
            batch_size: 批次大小
        """
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[ChunkResult],
        top_k: int = 10,
    ) -> List[ChunkResult]:
        """对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选ChunkResult列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的ChunkResult列表
        """
        if not candidates:
            return []

        # 构建查询-文档对
        pairs = [(query, c.text) for c in candidates]

        # 批量预测相关性分数
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # 将分数与候选结果绑定并排序
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, candidate in scored_candidates[:top_k]:
            results.append(ChunkResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                metadata=candidate.metadata,
            ))
        return results
