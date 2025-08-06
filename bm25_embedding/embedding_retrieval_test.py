#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding检索测试用例
使用向量嵌入和余弦相似度进行文档检索
"""

import numpy as np
from typing import List, Dict, Tuple
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class EmbeddingRetriever:
    """
    基于向量嵌入的检索器
    使用TF-IDF向量化和余弦相似度进行文档检索
    """
    
    def __init__(self, max_features: int = 1000, min_df: int = 1, max_df: float = 0.95):
        """
        初始化Embedding检索器
        
        Args:
            max_features: 最大特征数量
            min_df: 最小文档频率
            max_df: 最大文档频率比例
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理：分词、去除标点符号
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本字符串
        """
        # 去除标点符号和特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 中文分词
        words = jieba.lcut(text.lower())
        
        # 过滤空字符串和单字符，返回空格分隔的字符串
        words = [word.strip() for word in words if len(word.strip()) > 1]
        
        return ' '.join(words)
    
    def build_index(self, documents: List[str]):
        """
        构建向量索引
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        
        # 预处理所有文档
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r'\b\w+\b'  # 支持中文词汇
        )
        
        # 构建文档向量矩阵
        self.doc_vectors = self.vectorizer.fit_transform(processed_docs)
        
        print(f"📊 向量维度: {self.doc_vectors.shape}")
        print(f"📝 词汇表大小: {len(self.vectorizer.vocabulary_)}")
    
    def get_query_vector(self, query: str) -> np.ndarray:
        """
        将查询转换为向量
        
        Args:
            query: 查询字符串
            
        Returns:
            查询向量
        """
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        return query_vector
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        执行向量检索
        
        Args:
            query: 查询字符串
            top_k: 返回前k个结果
            
        Returns:
            (文档索引, 相似度分数, 文档内容)的列表，按相似度降序排列
        """
        # 获取查询向量
        query_vector = self.get_query_vector(query)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # 获取排序后的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 构建结果列表
        results = []
        for i in sorted_indices[:top_k]:
            results.append((i, similarities[i], self.documents[i]))
        
        return results
    
    def get_feature_weights(self, query: str, top_features: int = 10) -> List[Tuple[str, float]]:
        """
        获取查询中权重最高的特征词
        
        Args:
            query: 查询字符串
            top_features: 返回前N个特征
            
        Returns:
            (特征词, 权重)的列表
        """
        query_vector = self.get_query_vector(query)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 获取非零特征及其权重
        feature_weights = []
        for idx in query_vector.nonzero()[1]:
            weight = query_vector[0, idx]
            feature_weights.append((feature_names[idx], weight))
        
        # 按权重降序排序
        feature_weights.sort(key=lambda x: x[1], reverse=True)
        
        return feature_weights[:top_features]

class SimpleWordEmbeddingRetriever:
    """
    简单的词向量检索器
    使用词频向量和余弦相似度
    """
    
    def __init__(self):
        self.vocabulary = set()
        self.documents = []
        self.doc_vectors = []
        self.vocab_to_idx = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理
        """
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        words = jieba.lcut(text.lower())
        return [word.strip() for word in words if len(word.strip()) > 1]
    
    def build_index(self, documents: List[str]):
        """
        构建简单词向量索引
        """
        self.documents = documents
        
        # 构建词汇表
        for doc in documents:
            words = self.preprocess_text(doc)
            self.vocabulary.update(words)
        
        self.vocab_to_idx = {word: idx for idx, word in enumerate(sorted(self.vocabulary))}
        vocab_size = len(self.vocabulary)
        
        # 构建文档向量
        for doc in documents:
            words = self.preprocess_text(doc)
            word_counts = Counter(words)
            
            # 创建词频向量
            vector = np.zeros(vocab_size)
            for word, count in word_counts.items():
                if word in self.vocab_to_idx:
                    vector[self.vocab_to_idx[word]] = count
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self.doc_vectors.append(vector)
        
        self.doc_vectors = np.array(self.doc_vectors)
        print(f"📊 简单向量维度: {self.doc_vectors.shape}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        执行简单向量检索
        """
        words = self.preprocess_text(query)
        word_counts = Counter(words)
        
        # 创建查询向量
        query_vector = np.zeros(len(self.vocabulary))
        for word, count in word_counts.items():
            if word in self.vocab_to_idx:
                query_vector[self.vocab_to_idx[word]] = count
        
        # L2归一化
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # 计算余弦相似度
        similarities = np.dot(self.doc_vectors, query_vector)
        
        # 排序并返回结果
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in sorted_indices[:top_k]:
            results.append((i, similarities[i], self.documents[i]))
        
        return results

def test_embedding_retrieval():
    """
    Embedding检索测试用例
    """
    print("🧠 Embedding检索测试用例")
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
    
    # 测试TF-IDF向量检索
    print("\n🔍 TF-IDF向量检索测试")
    print("-" * 40)
    
    tfidf_retriever = EmbeddingRetriever(max_features=500)
    tfidf_retriever.build_index(documents)
    
    test_queries = [
        "Python编程语言",
        "机器学习算法",
        "深度学习神经网络",
        "数据科学分析",
        "信息检索技术"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 30)
        
        results = tfidf_retriever.search(query, top_k=3)
        
        for rank, (doc_idx, score, doc_content) in enumerate(results, 1):
            print(f"排名 {rank}: (相似度: {score:.4f})")
            print(f"文档 {doc_idx}: {doc_content}")
            print()
        
        # 显示查询特征权重
        feature_weights = tfidf_retriever.get_feature_weights(query, top_features=5)
        if feature_weights:
            print("🔑 关键特征词:")
            for feature, weight in feature_weights:
                print(f"  {feature}: {weight:.4f}")
            print()
    
    # 测试简单词向量检索
    print("\n🔍 简单词向量检索测试")
    print("-" * 40)
    
    simple_retriever = SimpleWordEmbeddingRetriever()
    simple_retriever.build_index(documents)
    
    query = "Python数据科学"
    print(f"\n查询: {query}")
    print("-" * 30)
    
    results = simple_retriever.search(query, top_k=3)
    
    for rank, (doc_idx, score, doc_content) in enumerate(results, 1):
        print(f"排名 {rank}: (相似度: {score:.4f})")
        print(f"文档 {doc_idx}: {doc_content}")
        print()
    
    # 性能对比测试
    print("\n⚡ 性能对比测试")
    print("-" * 40)
    
    import time
    
    test_query = "Python机器学习"
    
    # TF-IDF检索性能
    start_time = time.time()
    for _ in range(100):
        tfidf_results = tfidf_retriever.search(test_query, top_k=5)
    tfidf_time = (time.time() - start_time) / 100 * 1000
    
    # 简单向量检索性能
    start_time = time.time()
    for _ in range(100):
        simple_results = simple_retriever.search(test_query, top_k=5)
    simple_time = (time.time() - start_time) / 100 * 1000
    
    print(f"TF-IDF检索平均时间: {tfidf_time:.2f} ms")
    print(f"简单向量检索平均时间: {simple_time:.2f} ms")
    
    # 结果质量对比
    print("\n📊 结果质量对比")
    print("-" * 40)
    
    print(f"\n查询: {test_query}")
    print("\nTF-IDF结果:")
    for rank, (doc_idx, score, doc_content) in enumerate(tfidf_results[:3], 1):
        print(f"  {rank}. (分数: {score:.4f}) {doc_content[:50]}...")
    
    print("\n简单向量结果:")
    for rank, (doc_idx, score, doc_content) in enumerate(simple_results[:3], 1):
        print(f"  {rank}. (分数: {score:.4f}) {doc_content[:50]}...")
    
    # 向量空间可视化信息
    print("\n📈 向量空间信息")
    print("-" * 40)
    
    print(f"TF-IDF向量维度: {tfidf_retriever.doc_vectors.shape}")
    print(f"TF-IDF稀疏度: {1 - tfidf_retriever.doc_vectors.nnz / (tfidf_retriever.doc_vectors.shape[0] * tfidf_retriever.doc_vectors.shape[1]):.4f}")
    print(f"简单向量维度: {simple_retriever.doc_vectors.shape}")
    print(f"词汇表大小: {len(simple_retriever.vocabulary)}")

def compare_retrieval_methods():
    """
    比较不同检索方法的效果
    """
    print("\n🔄 检索方法对比分析")
    print("=" * 50)
    
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
        "机器学习是人工智能的一个重要分支，包括监督学习和无监督学习。",
        "深度学习是机器学习的子集，使用神经网络进行模式识别。",
        "自然语言处理是计算机科学和人工智能的交叉领域。",
        "数据科学结合了统计学、计算机科学和领域专业知识。"
    ]
    
    # 初始化检索器
    tfidf_retriever = EmbeddingRetriever(max_features=100)
    tfidf_retriever.build_index(documents)
    
    simple_retriever = SimpleWordEmbeddingRetriever()
    simple_retriever.build_index(documents)
    
    test_queries = [
        "Python编程",
        "机器学习人工智能",
        "数据科学统计"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 30)
        
        # TF-IDF结果
        tfidf_results = tfidf_retriever.search(query, top_k=2)
        print("TF-IDF Top-2:")
        for rank, (doc_idx, score, _) in enumerate(tfidf_results, 1):
            print(f"  {rank}. 文档{doc_idx} (分数: {score:.4f})")
        
        # 简单向量结果
        simple_results = simple_retriever.search(query, top_k=2)
        print("\n简单向量 Top-2:")
        for rank, (doc_idx, score, _) in enumerate(simple_results, 1):
            print(f"  {rank}. 文档{doc_idx} (分数: {score:.4f})")
        
        print()

if __name__ == "__main__":
    # 检查依赖
    try:
        import jieba
        import sklearn
        import numpy as np
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请安装: pip install jieba scikit-learn numpy")
        exit(1)
    
    test_embedding_retrieval()
    compare_retrieval_methods()