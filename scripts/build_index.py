"""构建索引脚本
使用方法：python scripts/build_index.py [--data-path PATH] [--index-dir DIR] [--config CONFIG]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """主函数：解析参数并构建索引"""
    parser = argparse.ArgumentParser(
        description="为GraphRAG出行知识问答系统构建向量索引和图谱索引",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/sample_travel_rules.md",
        help="原始数据文件路径（Markdown格式）",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed",
        help="索引保存目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    print(f"GraphRAG出行知识问答系统 - 索引构建工具")
    print(f"=" * 50)
    print(f"数据文件：{args.data_path}")
    print(f"索引目录：{args.index_dir}")
    print(f"配置文件：{args.config}")
    print()

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误：数据文件不存在：{args.data_path}")
        sys.exit(1)

    # 初始化流水线
    print("初始化流水线...")
    try:
        from src.pipeline.rag_pipeline import GraphRAGPipeline
        pipeline = GraphRAGPipeline(config_path=args.config)
    except Exception as e:
        print(f"初始化失败：{e}")
        sys.exit(1)

    # 构建索引
    print("开始构建索引（可能需要几分钟）...")
    start_time = time.time()

    try:
        stats = pipeline.build_indices(
            data_path=args.data_path,
            index_dir=args.index_dir,
        )
        elapsed = time.time() - start_time

        print(f"\n✓ 索引构建完成！")
        print(f"  文本块数量：{stats['chunk_count']}")
        print(f"  图节点数量：{stats['node_count']}")
        print(f"  耗时：{elapsed:.1f}秒")
        print(f"  索引保存位置：{args.index_dir}")
    except Exception as e:
        print(f"✗ 索引构建失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
