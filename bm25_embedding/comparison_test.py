#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25 vs Embedding检索对比测试
直接比较两种检索方法在相同数据集上的表现
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from bm25_retrieval_test import BM25Retriever
from embedding_retrieval_test import EmbeddingRetriever

class RetrievalComparison:
    """
    检索方法对比分析器
    """
    
    def __init__(self):
        self.bm25_retriever = BM25Retriever(k1=1.5, b=0.75)
        self.embedding_retriever = EmbeddingRetriever(max_features=1000)
        self.documents = []
        
    def load_test_data(self) -> List[str]:
        """
        加载测试数据集
        
        Returns:
            文档列表
        """
        documents = [
            "Python是一种高级编程语言，具有简洁的语法和强大的功能，广泛应用于Web开发、数据科学、人工智能等领域。",
            "机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习模式，包括监督学习、无监督学习和强化学习。",
            "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑处理信息的方式，在图像识别、语音识别等领域取得突破。",
            "自然语言处理结合了计算机科学、人工智能和语言学，目标是让计算机理解、处理和生成人类语言。",
            "数据科学是一个跨学科领域，结合统计学、计算机科学和领域专业知识，从数据中提取有价值的洞察。",
            "Python在数据分析领域有丰富的生态系统，包括NumPy、Pandas、Matplotlib等优秀的数据处理和可视化库。",
            "TensorFlow和PyTorch是目前最流行的深度学习框架，为研究人员和开发者提供了强大的工具。",
            "搜索引擎是信息检索系统的典型应用，使用复杂的算法来理解查询意图并返回相关结果。",
            "BM25算法是信息检索中的经典方法，基于概率模型，考虑词频和文档长度等因素计算相关性。",
            "向量空间模型将文档和查询表示为高维向量，通过计算向量间的相似度来衡量文档相关性。",
            "词嵌入技术将词汇映射到连续向量空间，捕获词汇间的语义关系，是现代NLP的基础技术。",
            "信息检索评估指标包括精确率、召回率、F1分数等，用于衡量检索系统的性能表现。",
            "倒排索引是搜索引擎的核心数据结构，将每个词映射到包含该词的文档列表，大大提高检索效率。",
            "文本预处理包括分词、去停用词、词干提取等步骤，是文本挖掘和信息检索的重要环节。",
            "相关性反馈是改进检索效果的重要技术，通过用户的反馈信息来调整查询或重新排序结果。"
        ]
        
        self.documents = documents
        return documents
    
    def build_indexes(self):
        """
        为两种检索方法构建索引
        """
        print("🔧 构建检索索引...")
        
        # 构建BM25索引
        start_time = time.time()
        self.bm25_retriever.build_index(self.documents)
        bm25_build_time = time.time() - start_time
        
        # 构建Embedding索引
        start_time = time.time()
        self.embedding_retriever.build_index(self.documents)
        embedding_build_time = time.time() - start_time
        
        print(f"✅ BM25索引构建时间: {bm25_build_time:.4f}秒")
        print(f"✅ Embedding索引构建时间: {embedding_build_time:.4f}秒")
        
        return bm25_build_time, embedding_build_time
    
    def compare_search_results(self, query: str, top_k: int = 5) -> Dict:
        """
        比较两种方法的搜索结果
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            
        Returns:
            包含两种方法结果的字典
        """
        # BM25检索
        start_time = time.time()
        bm25_results = self.bm25_retriever.search(query, top_k)
        bm25_time = time.time() - start_time
        
        # Embedding检索
        start_time = time.time()
        embedding_results = self.embedding_retriever.search(query, top_k)
        embedding_time = time.time() - start_time
        
        return {
            'query': query,
            'bm25': {
                'results': bm25_results,
                'time': bm25_time
            },
            'embedding': {
                'results': embedding_results,
                'time': embedding_time
            }
        }
    
    def calculate_overlap(self, results1: List[Tuple], results2: List[Tuple]) -> float:
        """
        计算两个结果列表的重叠度
        
        Args:
            results1: 第一个结果列表
            results2: 第二个结果列表
            
        Returns:
            重叠度（0-1之间）
        """
        docs1 = set([r[0] for r in results1])
        docs2 = set([r[0] for r in results2])
        
        if len(docs1) == 0 and len(docs2) == 0:
            return 1.0
        
        intersection = len(docs1.intersection(docs2))
        union = len(docs1.union(docs2))
        
        return intersection / union if union > 0 else 0.0
    
    def run_comprehensive_test(self):
        """
        运行综合对比测试
        """
        print("🚀 BM25 vs Embedding 综合对比测试")
        print("=" * 60)
        
        # 加载数据并构建索引
        self.load_test_data()
        bm25_build_time, embedding_build_time = self.build_indexes()
        
        # 测试查询集合
        test_queries = [
            "Python编程语言开发",
            "机器学习人工智能算法",
            "深度学习神经网络",
            "自然语言处理技术",
            "数据科学分析统计",
            "搜索引擎信息检索",
            "向量空间模型相似度",
            "文本挖掘预处理"
        ]
        
        print(f"\n📊 测试数据集: {len(self.documents)}个文档")
        print(f"🔍 测试查询: {len(test_queries)}个查询")
        print("\n" + "=" * 60)
        
        # 存储所有结果用于统计分析
        all_results = []
        bm25_times = []
        embedding_times = []
        overlaps = []
        
        # 逐个测试查询
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 查询 {i}: {query}")
            print("-" * 50)
            
            comparison = self.compare_search_results(query, top_k=5)
            all_results.append(comparison)
            
            bm25_times.append(comparison['bm25']['time'])
            embedding_times.append(comparison['embedding']['time'])
            
            # 计算结果重叠度
            overlap = self.calculate_overlap(
                comparison['bm25']['results'],
                comparison['embedding']['results']
            )
            overlaps.append(overlap)
            
            # 显示Top-3结果
            print("\n🏆 BM25 Top-3:")
            for rank, (doc_idx, score, doc_content) in enumerate(comparison['bm25']['results'][:3], 1):
                print(f"  {rank}. 文档{doc_idx:2d} (分数: {score:6.4f}) {doc_content[:60]}...")
            
            print("\n🧠 Embedding Top-3:")
            for rank, (doc_idx, score, doc_content) in enumerate(comparison['embedding']['results'][:3], 1):
                print(f"  {rank}. 文档{doc_idx:2d} (分数: {score:6.4f}) {doc_content[:60]}...")
            
            print(f"\n⏱️  检索时间 - BM25: {comparison['bm25']['time']*1000:.2f}ms, Embedding: {comparison['embedding']['time']*1000:.2f}ms")
            print(f"🔄 结果重叠度: {overlap:.2f}")
        
        # 统计分析
        print("\n" + "=" * 60)
        print("📈 统计分析结果")
        print("=" * 60)
        
        print(f"\n⚡ 性能对比:")
        print(f"  索引构建时间:")
        print(f"    BM25:     {bm25_build_time:.4f}秒")
        print(f"    Embedding: {embedding_build_time:.4f}秒")
        
        print(f"\n  平均检索时间:")
        print(f"    BM25:     {np.mean(bm25_times)*1000:.2f}ms (±{np.std(bm25_times)*1000:.2f}ms)")
        print(f"    Embedding: {np.mean(embedding_times)*1000:.2f}ms (±{np.std(embedding_times)*1000:.2f}ms)")
        
        print(f"\n🔄 结果一致性:")
        print(f"  平均重叠度: {np.mean(overlaps):.3f} (±{np.std(overlaps):.3f})")
        print(f"  最高重叠度: {np.max(overlaps):.3f}")
        print(f"  最低重叠度: {np.min(overlaps):.3f}")
        
        # 方法特点分析
        print(f"\n🔍 方法特点分析:")
        print(f"\n  BM25算法:")
        print(f"    ✅ 基于精确词匹配，解释性强")
        print(f"    ✅ 计算速度快，内存占用少")
        print(f"    ✅ 对词频敏感，适合关键词检索")
        print(f"    ❌ 无法处理语义相似性")
        print(f"    ❌ 对同义词和相关词不敏感")
        
        print(f"\n  Embedding方法:")
        print(f"    ✅ 能捕获语义相似性")
        print(f"    ✅ 对同义词和相关概念敏感")
        print(f"    ✅ 适合语义检索任务")
        print(f"    ❌ 计算复杂度较高")
        print(f"    ❌ 需要更多内存存储向量")
        
        # 推荐使用场景
        print(f"\n💡 推荐使用场景:")
        print(f"\n  选择BM25当:")
        print(f"    • 需要精确关键词匹配")
        print(f"    • 对性能要求极高")
        print(f"    • 文档集合较大，资源有限")
        print(f"    • 查询主要是具体的术语或名称")
        
        print(f"\n  选择Embedding当:")
        print(f"    • 需要语义理解和相似性匹配")
        print(f"    • 用户查询可能使用同义词")
        print(f"    • 文档内容丰富，概念性强")
        print(f"    • 可以接受稍高的计算成本")
        
        return all_results

def main():
    """
    主函数
    """
    try:
        # 检查依赖
        import jieba
        import sklearn
        import numpy as np
        
        # 运行对比测试
        comparison = RetrievalComparison()
        results = comparison.run_comprehensive_test()
        
        print("\n✅ 对比测试完成！")
        
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请安装依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()