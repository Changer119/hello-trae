#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25检索测试用例
使用BM25算法进行文档检索和相关性排序
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import jieba
import re

class BM25Retriever:
    """
    BM25检索器实现
    BM25是一种基于词频的检索算法，广泛用于信息检索系统
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: 控制词频饱和度的参数，通常取值1.2-2.0
            b: 控制文档长度归一化的参数，通常取值0.75
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理：分词、去除标点符号、转小写
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的词汇列表
        """
        # 去除标点符号和特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 中文分词
        words = jieba.lcut(text.lower())
        
        # 过滤空字符串和单字符
        words = [word.strip() for word in words if len(word.strip()) > 1]
        
        return words
    
    def build_index(self, documents: List[str]):
        """
        构建BM25索引
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_freqs = []
        
        # 预处理所有文档
        processed_docs = []
        for doc in documents:
            words = self.preprocess_text(doc)
            processed_docs.append(words)
            self.doc_freqs.append(Counter(words))
            self.doc_len.append(len(words))
        
        # 计算平均文档长度
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # 计算IDF值
        df = defaultdict(int)
        for doc_freq in self.doc_freqs:
            for word in doc_freq.keys():
                df[word] += 1
        
        for word, freq in df.items():
            self.idf[word] = math.log((len(documents) - freq + 0.5) / (freq + 0.5))
    
    def get_score(self, query: str, doc_idx: int) -> float:
        """
        计算查询与文档的BM25相关性分数
        
        Args:
            query: 查询字符串
            doc_idx: 文档索引
            
        Returns:
            BM25分数
        """
        query_words = self.preprocess_text(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0
        for word in query_words:
            if word in doc_freq:
                tf = doc_freq[word]
                idf = self.idf.get(word, 0)
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        执行BM25检索
        
        Args:
            query: 查询字符串
            top_k: 返回前k个结果
            
        Returns:
            (文档索引, 分数, 文档内容)的列表，按分数降序排列
        """
        scores = []
        for i in range(len(self.documents)):
            score = self.get_score(query, i)
            scores.append((i, score, self.documents[i]))
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]

def test_bm25_retrieval():
    """
    BM25检索测试用例
    """
    print("🔍 BM25检索测试用例")
    print("=" * 50)
    
    # 测试文档集合
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
        "机器学习是人工智能的一个重要分支，包括监督学习和无监督学习。",
        "深度学习是机器学习的子集，使用神经网络进行模式识别。",
        "自然语言处理是计算机科学和人工智能的交叉领域。",
        "数据科学结合了统计学、计算机科学和领域专业知识。",
        "Python在数据分析和可视化方面有很多优秀的库。",
        "TensorFlow和PyTorch是流行的深度学习框架。",
        "搜索引擎使用信息检索技术来查找相关文档。",
        "BM25是一种经典的文档检索算法，基于概率模型。",
        "向量空间模型是信息检索中的另一种重要方法。"
    ]
    
    # 创建BM25检索器
    retriever = BM25Retriever(k1=1.5, b=0.75)
    
    print("📚 构建文档索引...")
    retriever.build_index(documents)
    print(f"✅ 索引构建完成，共{len(documents)}个文档")
    
    # 测试查询
    test_queries = [
        "Python编程语言",
        "机器学习算法",
        "深度学习神经网络",
        "数据科学分析",
        "信息检索技术"
    ]
    
    print("\n🔍 开始检索测试...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 30)
        
        results = retriever.search(query, top_k=3)
        
        for rank, (doc_idx, score, doc_content) in enumerate(results, 1):
            print(f"排名 {rank}: (分数: {score:.4f})")
            print(f"文档 {doc_idx}: {doc_content}")
            print()
    
    # 性能测试
    print("\n⚡ 性能测试")
    print("-" * 30)
    
    import time
    
    query = "Python机器学习"
    start_time = time.time()
    
    for _ in range(100):
        results = retriever.search(query, top_k=5)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000
    
    print(f"平均检索时间: {avg_time:.2f} ms")
    print(f"检索吞吐量: {1000/avg_time:.0f} 查询/秒")
    
    # 算法参数影响测试
    print("\n🔧 参数影响测试")
    print("-" * 30)
    
    test_query = "Python数据科学"
    
    # 测试不同k1值
    print("\n不同k1值的影响:")
    for k1 in [0.5, 1.2, 1.5, 2.0]:
        test_retriever = BM25Retriever(k1=k1, b=0.75)
        test_retriever.build_index(documents)
        results = test_retriever.search(test_query, top_k=1)
        if results:
            print(f"k1={k1}: 最高分数={results[0][1]:.4f}")
    
    # 测试不同b值
    print("\n不同b值的影响:")
    for b in [0.0, 0.25, 0.75, 1.0]:
        test_retriever = BM25Retriever(k1=1.5, b=b)
        test_retriever.build_index(documents)
        results = test_retriever.search(test_query, top_k=1)
        if results:
            print(f"b={b}: 最高分数={results[0][1]:.4f}")

if __name__ == "__main__":
    # 确保jieba已安装
    try:
        import jieba
    except ImportError:
        print("❌ 请安装jieba: pip install jieba")
        exit(1)
    
    test_bm25_retrieval()