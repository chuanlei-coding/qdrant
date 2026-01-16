# pgvector SQL 能力说明

## 概述

pgvector 是 PostgreSQL 的一个扩展，用于在 PostgreSQL 中存储和搜索向量数据。它提供了新的数据类型、操作符、函数和索引类型，使 PostgreSQL 能够高效地处理向量相似度搜索。

---

## 一、数据类型

### 1.1 vector 类型

**说明**: `vector` 是 pgvector 提供的核心数据类型，用于存储密集向量数据。

**语法**:
```sql
vector(n)  -- n 是向量维度
```

**示例**:
```sql
-- 创建包含向量列的表
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding vector(1536),  -- 1536 维向量
    name TEXT
);

-- 插入向量数据
INSERT INTO items (name, embedding) VALUES
    ('item1', '[0.1, 0.2, 0.3, ...]'),
    ('item2', '[0.4, 0.5, 0.6, ...]');
```

**特点**:
- 维度必须指定，且在表创建时确定
- 向量数据以数组形式存储
- 支持高维向量（理论无限制，但实际受 PostgreSQL 行大小限制）

### 1.2 sparsevec 类型（pgvector 0.8.0+）

**说明**: `sparsevec` 是 pgvector 0.8.0 引入的新类型，用于存储稀疏向量（大多数维度为零的向量）。

**语法**:
```sql
sparsevec  -- 稀疏向量类型
```

**示例**:
```sql
-- 创建包含稀疏向量列的表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    embedding vector(1536),  -- 密集向量
    sparse_embedding sparsevec,  -- 稀疏向量
    name TEXT
);

-- 从数组转换为稀疏向量
ALTER TABLE documents ADD COLUMN sparse_embedding sparsevec;

UPDATE documents
SET sparse_embedding = my_real_array::sparsevec;

-- 使用稀疏向量进行查询
SELECT id, name, sparse_embedding <-> sparsevec '[0.0, 0.0, 1.0, 0.0, ...]' AS distance
FROM documents
ORDER BY sparse_embedding <-> sparsevec '[0.0, 0.0, 1.0, 0.0, ...]'
LIMIT 10;
```

**特点**:
- 适用于大多数维度为零的稀疏向量
- 节省存储空间
- 加速计算（只计算非零维度）
- 支持与 `vector` 类型相同的操作符

---

## 二、向量相似度操作符

### 2.1 余弦距离：`<=>` 

**说明**: 计算两个向量之间的余弦距离。值越小表示越相似（距离越小）。

**语法**:
```sql
vector <=> vector  -- 返回 float8
```

**示例**:
```sql
-- 查找最相似的向量（使用余弦距离）
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;

-- 在 WHERE 子句中使用
SELECT id, name
FROM items
WHERE embedding <=> '[0.1, 0.2, 0.3, ...]' < 0.3;
```

**返回值范围**: 0.0 - 2.0
- `0.0`: 完全相同
- `1.0`: 正交（不相关）
- `2.0`: 完全相反

### 2.2 内积（负）：`<#>` 

**说明**: 计算两个向量的负内积（负点积）。用于寻找相似向量时，值越小表示越相似。

**语法**:
```sql
vector <#> vector  -- 返回 float8
```

**示例**:
```sql
-- 查找最相似的向量（使用负内积）
SELECT id, name, embedding <#> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <#> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;
```

**注意**:
- 内积操作符返回的是**负内积**，所以 ORDER BY 时值越小越好
- 通常用于归一化后的向量（L2 归一化）

### 2.3 欧氏距离（L2）：`<->` 

**说明**: 计算两个向量之间的欧氏距离（L2 距离）。值越小表示越相似。

**语法**:
```sql
vector <-> vector  -- 返回 float8
```

**示例**:
```sql
-- 查找最相似的向量（使用欧氏距离）
SELECT id, name, embedding <-> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <-> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;

-- 在 WHERE 子句中使用
SELECT id, name
FROM items
WHERE embedding <-> '[0.1, 0.2, 0.3, ...]' < 0.5;
```

**返回值范围**: 0.0 - 无穷大
- `0.0`: 完全相同
- 值越大：越不相似

---

## 三、向量距离函数

### 3.1 cosine_distance()

**说明**: 计算两个向量之间的余弦距离（等价于 `<=>` 操作符）。

**语法**:
```sql
cosine_distance(vector1, vector2)  -- 返回 float8
```

**示例**:
```sql
SELECT id, name, cosine_distance(embedding, '[0.1, 0.2, 0.3, ...]') AS distance
FROM items
ORDER BY distance
LIMIT 10;
```

### 3.2 inner_product()

**说明**: 计算两个向量的内积（点积）。

**语法**:
```sql
inner_product(vector1, vector2)  -- 返回 float8
```

**示例**:
```sql
SELECT id, name, inner_product(embedding, '[0.1, 0.2, 0.3, ...]') AS similarity
FROM items
ORDER BY similarity DESC  -- 内积越大越好
LIMIT 10;
```

### 3.3 l2_distance()

**说明**: 计算两个向量之间的欧氏距离（等价于 `<->` 操作符）。

**语法**:
```sql
l2_distance(vector1, vector2)  -- 返回 float8
```

**示例**:
```sql
SELECT id, name, l2_distance(embedding, '[0.1, 0.2, 0.3, ...]') AS distance
FROM items
ORDER BY distance
LIMIT 10;
```

---

## 四、向量索引

### 4.1 IVFFlat 索引

**说明**: Inverted File with Flat Compression，适用于大规模数据集的近似最近邻搜索。

**创建索引**:
```sql
-- 创建 IVFFlat 索引
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 参数说明：
-- - lists: 聚类中心数量（建议为行数的 1/1000 到 1/10）
-- - 操作符类：
--   - vector_cosine_ops: 用于余弦距离（<=>）
--   - vector_l2_ops: 用于欧氏距离（<->）
--   - vector_ip_ops: 用于内积（<#>）
```

**示例**:
```sql
-- 使用余弦距离索引
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 使用欧氏距离索引
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- 使用内积索引
CREATE INDEX ON items USING ivfflat (embedding vector_ip_ops)
WITH (lists = 100);
```

**特点**:
- 查询速度较快
- 索引构建时间较短
- 索引大小较小
- 精确度可调（通过 `lists` 参数）

### 4.2 HNSW 索引

**说明**: Hierarchical Navigable Small World，提供更高质量的近似最近邻搜索。

**创建索引**:
```sql
-- 创建 HNSW 索引
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 参数说明：
-- - m: 每个节点的最大连接数（默认 16，范围 4-64）
-- - ef_construction: 构建时的候选集大小（默认 64，范围 4-1000）
-- - 操作符类：同上
```

**示例**:
```sql
-- 使用余弦距离索引
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 使用欧氏距离索引
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- 使用内积索引
CREATE INDEX ON items USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);
```

**特点**:
- 查询速度非常快
- 精确度较高
- 索引构建时间较长
- 索引大小较大

### 4.3 索引选择建议

| 场景 | 推荐索引 | 原因 |
|------|---------|------|
| 大规模数据（百万级+） | IVFFlat | 构建快，索引小 |
| 高精度要求 | HNSW | 精确度更高 |
| 查询频率高 | HNSW | 查询速度快 |
| 内存受限 | IVFFlat | 索引占用小 |
| 写入频繁 | IVFFlat | 构建更快 |

---

## 五、高级特性（pgvector 0.8.0+）

### 5.1 智能过滤与索引估算

**说明**: pgvector 0.8.0 增强了过滤条件下执行向量索引或 B-tree 索引的选择策略。当查询包含 `WHERE` 子句时，PostgreSQL 会智能判断是否使用向量索引还是 B-tree 索引。

**示例**:
```sql
-- 假设有表 documents(embedding vector(1536), category text)
-- 创建 HNSW 索引
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- 创建 B-tree 索引用于过滤
CREATE INDEX ON documents (category);

-- 查询时，PostgreSQL 会智能选择索引
SELECT *
FROM documents
WHERE category = 'science'  -- 过滤条件
ORDER BY embedding <-> '[...]'  -- 向量搜索
LIMIT 10;

-- 如果 category 过滤后数据量很大，可能使用向量索引
-- 如果 category 过滤后数据量很小，可能先使用 B-tree 索引
```

**优势**:
- 根据过滤条件自动选择最优索引
- 在过滤后数据量大的情况下使用向量索引
- 在过滤后数据量小的情况下使用 B-tree 索引

### 5.2 迭代式索引扫描（Iterative Index Scans）

**说明**: pgvector 0.8.0 引入了迭代式索引扫描机制。当使用 HNSW 或 IVFFlat 索引查询，且原始索引扫描没有返回足够的条目来满足过滤条件时，可以继续向后扫描索引直到满足条件或达到阈值。

**配置参数**:
```sql
-- HNSW 迭代扫描
SET hnsw.iterative_scan = ON;
SET hnsw.max_scan_tuples = 5000;  -- 最大扫描元组数

-- IVFFlat 迭代扫描
SET ivfflat.iterative_scan = ON;
SET ivfflat.max_probes = 100;  -- 最大探测数
```

**示例**:
```sql
-- 开启迭代扫描
SET hnsw.iterative_scan = ON;
SET hnsw.max_scan_tuples = 5000;

-- 如果第一次检索结果中，符合 category 'science' 的记录数量太少，
-- 会自动继续检索直到找到足够的记录或达到阈值
SELECT *
FROM documents
WHERE category = 'science'
ORDER BY embedding <-> '[...]'
LIMIT 10;
```

**优势**:
- 防止因过滤过强导致返回结果过少
- 提高查询召回率
- 可配置扫描阈值

### 5.3 HNSW 并行构建（性能提升）

**说明**: pgvector 0.8.0 支持并行构建 HNSW 索引，显著减少建索引时间和 WAL 日志生成量。

**示例**:
```sql
-- 并行构建 HNSW 索引（PostgreSQL 自动选择并行度）
CREATE INDEX CONCURRENTLY ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 查看构建进度
SELECT * FROM pg_stat_progress_create_index;
```

**性能提升**:
- 构建时间显著减少（特别是大规模数据集）
- WAL 日志生成量减少
- 写入负载降低

### 5.4 查询性能优化

**说明**: pgvector 0.8.0 对大型数据集（百万级 embedding）的查询性能和资源使用进行了优化。

**示例**:
```sql
-- 对于百万级向量的查询，性能显著提升
SELECT id, name, embedding <=> '[...]' AS distance
FROM documents
ORDER BY embedding <=> '[...]'
LIMIT 10;

-- 结合过滤条件时，优化更明显
SELECT id, name, embedding <=> '[...]' AS distance
FROM documents
WHERE category = 'science'
ORDER BY embedding <=> '[...]'
LIMIT 10;
```

**性能指标**:
- 查询响应时间减少（特别是在大规模数据集上）
- 资源使用优化（CPU、内存）
- 索引扫描效率提升

---

## 六、查询示例

### 5.1 基本相似度搜索

```sql
-- 使用余弦距离查找最相似的 10 个向量
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;

-- 使用欧氏距离查找最相似的 10 个向量
SELECT id, name, embedding <-> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <-> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;
```

### 5.2 带距离阈值的搜索

```sql
-- 查找距离小于 0.3 的所有向量（余弦距离）
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE embedding <=> '[0.1, 0.2, 0.3, ...]' < 0.3
ORDER BY distance;

-- 查找距离小于 0.5 的所有向量（欧氏距离）
SELECT id, name, embedding <-> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE embedding <-> '[0.1, 0.2, 0.3, ...]' < 0.5
ORDER BY distance;
```

### 5.3 结合其他条件的搜索

```sql
-- 在特定类别中查找相似向量
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE category = 'electronics'
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;

-- 结合时间范围
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE created_at > '2024-01-01'
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;
```

### 5.4 使用子查询

```sql
-- 查找与某个特定项最相似的其他项
SELECT i2.id, i2.name, i1.embedding <=> i2.embedding AS distance
FROM items i1
CROSS JOIN items i2
WHERE i1.id = 123
  AND i2.id != i1.id
ORDER BY distance
LIMIT 10;

-- 使用 JOIN
SELECT i2.id, i2.name, i1.embedding <=> i2.embedding AS distance
FROM items i1
JOIN items i2 ON i1.id != i2.id
WHERE i1.id = 123
ORDER BY distance
LIMIT 10;
```

### 5.5 聚合查询

```sql
-- 计算每个类别中最相似的向量
SELECT 
    category,
    id,
    name,
    embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE (category, embedding <=> '[0.1, 0.2, 0.3, ...]') IN (
    SELECT category, MIN(embedding <=> '[0.1, 0.2, 0.3, ...]')
    FROM items
    GROUP BY category
);
```

---

## 七、其他高级特性

### 6.1 向量维度检查

```sql
-- 检查向量维度
SELECT vector_dims(embedding) AS dimensions
FROM items
LIMIT 1;

-- 创建表时验证维度
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding vector(1536) CHECK (vector_dims(embedding) = 1536),
    name TEXT
);
```

### 6.2 向量归一化

```sql
-- L2 归一化向量（在应用层或使用函数）
-- 注意：pgvector 本身不直接提供归一化函数，
-- 通常需要在应用层处理，或者使用自定义函数

-- 示例：使用自定义函数进行 L2 归一化
CREATE OR REPLACE FUNCTION l2_normalize(v vector) RETURNS vector AS $$
BEGIN
    RETURN v / sqrt(v <#> v);  -- 内积的平方根
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 使用
SELECT l2_normalize(embedding) AS normalized_embedding
FROM items;
```

### 6.3 批量插入和更新

```sql
-- 批量插入向量
INSERT INTO items (name, embedding) VALUES
    ('item1', '[0.1, 0.2, 0.3, ...]'),
    ('item2', '[0.4, 0.5, 0.6, ...]'),
    ('item3', '[0.7, 0.8, 0.9, ...]');

-- 批量更新向量
UPDATE items
SET embedding = '[0.1, 0.2, 0.3, ...]'
WHERE id IN (1, 2, 3);
```

### 6.4 使用索引提示

```sql
-- 在查询中使用索引（PostgreSQL 会自动使用合适的索引）
-- 但可以通过 EXPLAIN 查看执行计划

EXPLAIN ANALYZE
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;
```

---

## 八、性能优化

### 7.1 索引参数调优

```sql
-- IVFFlat 索引参数调优
-- lists: 建议为 rows / 1000 到 rows / 10
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW 索引参数调优
-- m: 越大查询越快，但索引越大（默认 16）
-- ef_construction: 越大索引质量越好，但构建越慢（默认 64）
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);
```

### 7.2 查询优化

```sql
-- 使用 LIMIT 限制结果数量
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;  -- 限制结果数量

-- 使用 WHERE 子句提前过滤
SELECT id, name, embedding <=> '[0.1, 0.2, 0.3, ...]' AS distance
FROM items
WHERE category = 'electronics'  -- 提前过滤
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 10;
```

### 7.3 索引维护

```sql
-- 重建索引
REINDEX INDEX items_embedding_idx;

-- 分析表（更新统计信息）
ANALYZE items;
```

---

## 九、完整示例

### 8.1 创建表和索引

```sql
-- 1. 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 创建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(1536),  -- 1536 维向量（如 OpenAI 的 text-embedding-ada-002）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 创建 HNSW 索引（用于余弦距离搜索）
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. 插入示例数据
INSERT INTO documents (title, content, embedding) VALUES
    ('Document 1', 'This is the first document', '[0.1, 0.2, ...]'),
    ('Document 2', 'This is the second document', '[0.3, 0.4, ...]'),
    ('Document 3', 'This is the third document', '[0.5, 0.6, ...]');
```

### 8.2 执行相似度搜索

```sql
-- 搜索最相似的文档
SELECT 
    id,
    title,
    content,
    embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;

-- 搜索相似度高于阈值的文档
SELECT 
    id,
    title,
    content,
    embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
WHERE embedding <=> '[0.1, 0.2, ...]' < 0.3
ORDER BY distance;
```

### 8.3 混合搜索（向量 + 关键词）

```sql
-- 结合向量相似度和全文搜索
SELECT 
    id,
    title,
    content,
    embedding <=> '[0.1, 0.2, ...]' AS vector_distance,
    ts_rank(to_tsvector('english', content), query) AS text_rank
FROM documents,
     to_tsquery('english', 'search & terms') AS query
WHERE to_tsvector('english', content) @@ query
ORDER BY 
    (embedding <=> '[0.1, 0.2, ...]') * 0.7 +  -- 向量权重 70%
    (1 - ts_rank(to_tsvector('english', content), query)) * 0.3  -- 文本权重 30%
LIMIT 10;
```

---

## 十、注意事项

### 9.1 向量维度

- 向量维度必须在表创建时确定，且不能更改
- 所有向量的维度必须相同
- 维度选择取决于使用的嵌入模型（如 OpenAI: 1536, BGE: 768）

### 9.2 距离计算

- **余弦距离** (`<=>`): 适合大多数文本嵌入场景
- **欧氏距离** (`<->`): 适合连续数值向量
- **内积** (`<#>`): 适合 L2 归一化后的向量

### 9.3 索引选择

- **IVFFlat**: 适合大规模数据，构建速度快
- **HNSW**: 适合高精度要求，查询速度快
- 索引只能在表有数据后创建（因为需要计算聚类中心）

### 9.4 性能考虑

- 向量搜索通常比传统 SQL 查询慢，但索引可以显著提升性能
- 高维向量会占用更多存储空间
- 索引构建可能需要较长时间

---

## 十一、版本兼容性

### 10.1 PostgreSQL 版本要求

- pgvector 0.5.0+: PostgreSQL 11+
- pgvector 0.6.0+: PostgreSQL 12+（推荐）
- 最新版本: PostgreSQL 12-16

### 10.2 安装方式

```bash
# 使用 pgxn
pgxn install vector

# 使用 Homebrew (macOS)
brew install pgvector

# 从源码编译
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

### 10.3 启用扩展

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- 检查版本
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

---

## 十二、版本特性总结

### 12.1 pgvector 0.8.0 新特性

| 特性 | 说明 | 适用场景 |
|------|------|---------|
| **稀疏向量支持** | 新增 `sparsevec` 类型 | 稀疏高维向量（大多数维度为零） |
| **智能过滤** | 根据 `WHERE` 条件自动选择索引 | 带过滤条件的向量查询 |
| **迭代式索引扫描** | 防止过滤过强导致结果过少 | 混合向量搜索 + 过滤 |
| **并行构建 HNSW** | 支持并行构建索引 | 大规模向量集合，减少构建时间 |
| **查询性能优化** | 优化大型数据集查询性能 | 百万级向量查询 |

### 12.2 版本对比

| 版本 | 主要特性 |
|------|---------|
| **0.5.0** | 基础向量类型和操作符 |
| **0.6.0** | HNSW 索引支持 |
| **0.7.0** | 性能优化和 bug 修复 |
| **0.8.0** | 稀疏向量、智能过滤、迭代扫描、并行构建 |

---

## 十三、总结

### 11.1 核心能力

1. **向量数据类型**: `vector(n)` 用于存储向量
2. **相似度操作符**: `<=>`, `<#>`, `<->` 用于计算距离
3. **距离函数**: `cosine_distance()`, `inner_product()`, `l2_distance()`
4. **向量索引**: `IVFFlat` 和 `HNSW` 用于加速搜索

### 11.2 使用场景

- **语义搜索**: 基于向量相似度的文档搜索
- **推荐系统**: 查找相似项
- **图像搜索**: 基于图像嵌入的相似图像搜索
- **异常检测**: 查找异常向量

### 13.1 核心能力

1. **向量数据类型**: `vector(n)` 用于存储密集向量，`sparsevec` 用于存储稀疏向量
2. **相似度操作符**: `<=>`, `<#>`, `<->` 用于计算距离
3. **距离函数**: `cosine_distance()`, `inner_product()`, `l2_distance()`
4. **向量索引**: `IVFFlat` 和 `HNSW` 用于加速搜索
5. **智能过滤** (0.8.0+): 根据 `WHERE` 条件自动选择最优索引
6. **迭代式扫描** (0.8.0+): 防止过滤过强导致结果过少
7. **并行构建** (0.8.0+): 支持并行构建 HNSW 索引

### 13.2 使用场景

- **语义搜索**: 基于向量相似度的文档搜索
- **推荐系统**: 查找相似项
- **图像搜索**: 基于图像嵌入的相似图像搜索
- **异常检测**: 查找异常向量
- **稀疏数据**: 处理稀疏高维向量（0.8.0+）

### 13.3 优势

- **原生集成**: 作为 PostgreSQL 扩展，与现有 SQL 工作流无缝集成
- **ACID 保证**: 利用 PostgreSQL 的事务和一致性保证
- **灵活查询**: 可以结合传统 SQL 条件进行混合查询
- **高性能**: 通过索引提供高效的近似最近邻搜索
- **智能优化** (0.8.0+): 自动选择最优索引策略
- **稀疏支持** (0.8.0+): 支持稀疏向量，节省存储和计算资源

---

## 参考资料

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [pgvector 文档](https://github.com/pgvector/pgvector#readme)
- [PostgreSQL 官方文档](https://www.postgresql.org/docs/)
