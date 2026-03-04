"""置信度门控模块 - 评估答案质量并决定是否需要澄清"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.indexing.faiss_index import ChunkResult
from src.generation.evidence_constrained import GenerationResult


@dataclass
class GateResult:
    """门控评估结果数据类"""
    confidence: float
    should_clarify: bool
    clarification: Optional[str]
    answer: str
    warning: Optional[str]


class ConfidenceGate:
    """置信度门控器
    
    综合评估答案的置信度，包括证据覆盖率、证据一致性和检索质量，
    低置信度时建议用户澄清问题。
    """

    def __init__(
        self,
        threshold: float = 0.4,
        high_threshold: float = 0.7,
    ) -> None:
        """初始化置信度门控器
        
        Args:
            threshold: 低置信度阈值，低于此值建议澄清
            high_threshold: 高置信度阈值，低于此值添加警告
        """
        self.threshold = threshold
        self.high_threshold = high_threshold

    def evaluate(
        self,
        query: str,
        generation_result: GenerationResult,
        retrieval_results: List[ChunkResult],
    ) -> GateResult:
        """评估生成结果的置信度
        
        计算方式：
        - evidence_coverage：检索结果中被引用的比例
        - evidence_consistency：模型自报的置信度分数
        - retrieval_score：前3个检索结果的平均分数
        - final_confidence = 0.4 * evidence_consistency + 0.3 * evidence_coverage + 0.3 * retrieval_score
        
        Args:
            query: 用户问题
            generation_result: 生成结果
            retrieval_results: 检索结果列表
            
        Returns:
            GateResult评估结果
        """
        # 计算证据覆盖率
        if retrieval_results:
            cited_set = set(generation_result.cited_chunks)
            retrieved_ids = {r.chunk_id for r in retrieval_results}
            cited_retrieved = cited_set & retrieved_ids
            evidence_coverage = len(cited_retrieved) / len(retrieval_results)
        else:
            evidence_coverage = 0.0

        # 证据一致性（使用模型自报置信度）
        evidence_consistency = generation_result.confidence_score

        # 检索质量（前3个结果的平均分）
        if retrieval_results:
            top_3 = retrieval_results[:3]
            retrieval_score = sum(r.score for r in top_3) / len(top_3)
            # 归一化到[0,1]（FAISS内积分数已在[-1,1]，BM25归一化到[0,1]）
            retrieval_score = max(0.0, min(1.0, retrieval_score))
        else:
            retrieval_score = 0.0

        # 综合置信度
        final_confidence = (
            0.4 * evidence_consistency
            + 0.3 * evidence_coverage
            + 0.3 * retrieval_score
        )
        final_confidence = max(0.0, min(1.0, final_confidence))

        # 决定是否需要澄清
        should_clarify = final_confidence < self.threshold
        clarification: Optional[str] = None
        warning: Optional[str] = None

        if should_clarify:
            clarification = (
                f"当前答案置信度较低（{final_confidence:.2f}），建议您：\n"
                f"1. 提供更具体的问题描述\n"
                f"2. 补充相关背景信息（如航班日期、目的地等）\n"
                f"3. 尝试分解为更简单的子问题"
            )
        elif final_confidence < self.high_threshold:
            warning = (
                f"注意：当前答案置信度为 {final_confidence:.2f}，"
                f"建议验证关键信息的准确性。"
            )

        return GateResult(
            confidence=final_confidence,
            should_clarify=should_clarify,
            clarification=clarification,
            answer=generation_result.answer,
            warning=warning,
        )
