"""证据约束生成模块 - 基于检索证据的受控文本生成"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.indexing.faiss_index import ChunkResult


@dataclass
class GenerationResult:
    """生成结果数据类"""
    answer: str
    cited_chunks: List[str]
    confidence_score: float


class EvidenceConstrainedGenerator:
    """证据约束生成器
    
    使用LangChain ChatOpenAI，基于检索到的证据块生成答案。
    强制模型仅根据提供的证据作答，并引用证据来源，
    输出包含置信度分数的结构化JSON。
    """

    _SYSTEM_PROMPT = """你是一个专业的出行知识问答助手。你的任务是根据提供的证据块回答用户问题。

规则：
1. 只能根据提供的证据块内容回答，不得使用外部知识
2. 答案中必须引用相关的证据块ID（chunk_id）
3. 如果证据不足，诚实说明无法从现有证据中找到答案
4. 评估你的答案置信度（0.0-1.0）

你必须以以下JSON格式输出（不要添加任何其他内容）：
{
  "answer": "你的回答",
  "cited_chunks": ["chunk_0001", "chunk_0002"],
  "confidence_score": 0.85
}"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
    ) -> None:
        """初始化证据约束生成器
        
        Args:
            model_name: OpenAI模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            base_url: 可选的API基础URL
        """
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if base_url:
            kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**kwargs)

    def _build_evidence_text(self, evidence_chunks: List[ChunkResult]) -> str:
        """构建证据文本
        
        Args:
            evidence_chunks: 证据块列表
            
        Returns:
            格式化的证据文本
        """
        lines = []
        for i, chunk in enumerate(evidence_chunks, 1):
            lines.append(f"[证据{i}] ID: {chunk.chunk_id}")
            lines.append(f"内容: {chunk.text}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> GenerationResult:
        """解析LLM响应
        
        Args:
            response_text: LLM输出文本
            
        Returns:
            GenerationResult对象
        """
        # 尝试提取JSON
        # 先尝试直接解析
        try:
            data = json.loads(response_text.strip())
            return GenerationResult(
                answer=str(data.get("answer", "")),
                cited_chunks=list(data.get("cited_chunks", [])),
                confidence_score=float(data.get("confidence_score", 0.5)),
            )
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取JSON块
        json_pattern = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_pattern:
            try:
                data = json.loads(json_pattern.group())
                return GenerationResult(
                    answer=str(data.get("answer", response_text)),
                    cited_chunks=list(data.get("cited_chunks", [])),
                    confidence_score=float(data.get("confidence_score", 0.3)),
                )
            except json.JSONDecodeError:
                pass

        # 解析失败，返回原始文本
        return GenerationResult(
            answer=response_text,
            cited_chunks=[],
            confidence_score=0.3,
        )

    def generate(
        self,
        query: str,
        evidence_chunks: List[ChunkResult],
    ) -> GenerationResult:
        """基于证据生成答案
        
        Args:
            query: 用户问题
            evidence_chunks: 检索到的证据块列表
            
        Returns:
            GenerationResult对象
        """
        if not evidence_chunks:
            return GenerationResult(
                answer="抱歉，未找到相关证据，无法回答该问题。",
                cited_chunks=[],
                confidence_score=0.0,
            )

        evidence_text = self._build_evidence_text(evidence_chunks)
        user_message = f"""问题：{query}

以下是相关证据：

{evidence_text}

请根据以上证据回答问题，并以JSON格式输出。"""

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.invoke(messages)
            return self._parse_response(response.content)
        except Exception as e:
            return GenerationResult(
                answer=f"生成过程出错：{str(e)}",
                cited_chunks=[],
                confidence_score=0.0,
            )
