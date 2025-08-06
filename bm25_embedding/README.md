# BM25 vs Embedding 检索测试用例

本目录包含了两种常用信息检索方法的完整测试用例：BM25算法和Embedding向量检索。

## 📁 文件结构

```
bm25_embedding/
├── README.md                    # 说明文档
├── requirements.txt             # 依赖包列表
├── bm25_retrieval_test.py      # BM25检索测试用例
├── embedding_retrieval_test.py  # Embedding检索测试用例
└── comparison_test.py          # 两种方法对比测试
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd bm25_embedding
pip install -r requirements.txt
```

### 2. 运行测试用例

```bash
# 运行BM25检索测试
python bm25_retrieval_test.py

# 运行Embedding检索测试
python embedding_retrieval_test.py

# 运行对比测试
python comparison_test.py
```

## 📊 测试用例详解

### BM25检索测试 (`bm25_retrieval_test.py`)

**算法特点：**
- 基于概率模型的经典检索算法
- 考虑词频(TF)和逆文档频率(IDF)
- 包含文档长度归一化
- 计算速度快，内存占用少

**测试内容：**
- ✅ 基础BM25检索功能
- ✅ 参数调优测试（k1, b值影响）
- ✅ 性能基准测试
- ✅ 中文文本分词处理

**核心公式：**
```
BM25(q,d) = Σ IDF(qi) × (f(qi,d) × (k1+1)) / (f(qi,d) + k1×(1-b+b×|d|/avgdl))
```

### Embedding检索测试 (`embedding_retrieval_test.py`)

**算法特点：**
- 基于向量空间模型
- 使用TF-IDF向量化
- 通过余弦相似度计算相关性
- 能够捕获语义相似性

**测试内容：**
- ✅ TF-IDF向量检索
- ✅ 简单词频向量检索
- ✅ 特征权重分析
- ✅ 向量空间可视化信息
- ✅ 性能对比测试

**核心技术：**
- TF-IDF向量化
- 余弦相似度计算
- L2向量归一化

### 对比测试 (`comparison_test.py`)

**测试功能：**
- 🔄 同一数据集上的直接对比
- ⏱️ 性能基准测试（索引构建时间、检索时间）
- 📈 结果质量分析（重叠度、一致性）
- 💡 使用场景推荐

## 📋 测试结果示例

### BM25检索结果
```
查询: Python编程语言
排名 1: (分数: 2.1543) Python是一种高级编程语言，广泛用于数据科学...
排名 2: (分数: 1.8932) Python在数据分析和可视化方面有很多优秀的库...
```

### Embedding检索结果
```
查询: Python编程语言
排名 1: (相似度: 0.8765) Python是一种高级编程语言，广泛用于数据科学...
排名 2: (相似度: 0.7234) Python在数据分析和可视化方面有很多优秀的库...
```

## 🔍 算法对比分析

| 特性 | BM25 | Embedding |
|------|------|----------|
| **计算复杂度** | 低 | 中等 |
| **内存占用** | 少 | 较多 |
| **语义理解** | 无 | 有 |
| **精确匹配** | 强 | 弱 |
| **同义词处理** | 无 | 有 |
| **可解释性** | 高 | 中等 |
| **适用场景** | 关键词检索 | 语义检索 |

## 💡 使用建议

### 选择BM25当：
- ✅ 需要精确关键词匹配
- ✅ 对性能要求极高
- ✅ 文档集合较大，资源有限
- ✅ 查询主要是具体的术语或名称

### 选择Embedding当：
- ✅ 需要语义理解和相似性匹配
- ✅ 用户查询可能使用同义词
- ✅ 文档内容丰富，概念性强
- ✅ 可以接受稍高的计算成本

## 🛠️ 自定义测试

### 添加自己的文档集合

```python
# 在任意测试文件中修改documents列表
documents = [
    "你的文档1",
    "你的文档2",
    # ... 更多文档
]
```

### 调整算法参数

```python
# BM25参数调整
bm25_retriever = BM25Retriever(k1=1.5, b=0.75)

# Embedding参数调整
embedding_retriever = EmbeddingRetriever(
    max_features=1000,
    min_df=1,
    max_df=0.95
)
```

## 📚 扩展阅读

### BM25算法
- [Okapi BM25 - Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.

### 向量空间模型
- Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to information retrieval.

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这些测试用例！

## 📄 许可证

本项目采用MIT许可证。