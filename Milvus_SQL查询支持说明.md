# Milvus SQL 查询支持说明

## 概述

Milvus 是一个开源的向量数据库，专为大规模向量相似度搜索设计。虽然 Milvus **不直接支持标准 SQL**（如 `SELECT ... FROM ... JOIN ... GROUP BY` 的完整功能），但它提供了**类 SQL 的查询能力**，主要用于非向量字段（metadata/scalar fields）的筛选和向量检索结合使用。

---

## 一、Milvus 支持的类 SQL 能力

### 1.1 过滤表达式（Filter Expressions）

**说明**: Milvus 支持对 collection 的非向量字段进行筛选，即 metadata 或 scalar 字段。

**支持的表达式**:
- **比较运算符**: `>`, `<`, `==`, `!=`, `>=`, `<=`
- **字符串匹配**: `like`（模式匹配）
- **布尔组合**: `and`, `or`, `not`
- **数组操作**: `array_contains`, `array_length`
- **JSON 字段操作**: 支持 JSON 字段的查询

**示例（Python SDK）**:
```python
from pymilvus import Collection

collection = Collection("my_collection")

# 基础过滤查询
results = collection.query(
    expr="category == 'electronics' AND price > 100",
    output_fields=["id", "name", "category", "price"],
    limit=10
)

# 字符串匹配
results = collection.query(
    expr="name like 'iPhone%'",
    output_fields=["id", "name"],
    limit=10
)

# 范围查询
results = collection.query(
    expr="date >= '2024-01-01' AND date <= '2024-12-31'",
    output_fields=["id", "name", "date"],
    limit=10
)

# 数组字段查询
results = collection.query(
    expr="array_contains(tags, 'popular')",
    output_fields=["id", "name", "tags"],
    limit=10
)
```

### 1.2 Get 操作（按主键查询）

**说明**: 类似于 SQL 的 `SELECT * FROM table WHERE id IN (...)`。

**示例**:
```python
# 按主键查询（类似于 SQL: SELECT * FROM items WHERE id IN (1, 2, 3)）
results = collection.get(
    ids=[1, 2, 3],
    output_fields=["id", "name", "category"]
)
```

**特点**:
- 直接通过主键获取实体
- 不需要过滤表达式
- 性能最优（直接定位）

### 1.3 Query 操作（带过滤条件查询）

**说明**: 使用 filter 条件来筛选实体，类似于 SQL 的 `SELECT ... WHERE ... LIMIT ...`。

**示例**:
```python
# 查询特定类别的实体（类似于 SQL: SELECT * FROM items WHERE category = 'electronics' LIMIT 10）
results = collection.query(
    expr="category == 'electronics'",
    output_fields=["id", "name", "category", "price"],
    limit=10
)

# 结合多个条件
results = collection.query(
    expr="category == 'electronics' AND price > 100 AND price < 1000",
    output_fields=["id", "name", "category", "price"],
    limit=20
)
```

### 1.4 QueryIterator 操作（分页查询）

**说明**: 支持分页查询结果，类似于 SQL 的 `LIMIT ... OFFSET ...`（但 Milvus 限制 offset + limit 的范围）。

**示例**:
```python
# 分页查询（类似于 SQL: SELECT * FROM items WHERE color = 'blue' ORDER BY id LIMIT 20 OFFSET 40）
query_iterator = collection.query_iterator(
    expr="color == 'blue'",
    output_fields=["id", "name", "color"],
    batch_size=20
)

# 逐批获取结果
while True:
    batch = query_iterator.next()
    if not batch:
        break
    # 处理批次数据
    for item in batch:
        print(item)
```

**限制**:
- Milvus 对 offset + limit 的范围有限制
- 不支持真正的 `ORDER BY`（只能按主键或向量相似度排序）
- 分页需要客户端实现

### 1.5 混合搜索（Hybrid Search）

**说明**: 结合向量相似度搜索和 metadata 过滤条件，实现类似 SQL 的混合查询。

**示例**:
```python
# 混合搜索（类似于 SQL: 
# SELECT * FROM products 
# WHERE category = 'Electronics' AND date >= '2024-01-01' 
# ORDER BY similarity(vector, query_vector) 
# LIMIT 10
results = collection.search(
    data=[query_vector],  # 查询向量
    anns_field="embedding",  # 向量字段名
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    expr="category == 'Electronics' AND date >= '2024-01-01'",  # 过滤条件
    limit=10,
    output_fields=["id", "name", "category", "date"]
)
```

**特点**:
- 先应用 metadata 过滤条件
- 再在过滤后的结果中进行向量相似度搜索
- 返回最相似的前 N 个结果

---

## 二、Milvus 不支持的 SQL 能力

### 2.1 表之间的 JOIN

**不支持**: Milvus 中 collection 之间的 JOIN 不是核心功能。

**替代方案**:
- 在应用层实现 JOIN 逻辑
- 使用预聚合数据（在插入时合并数据）
- 使用其他数据库（如 PostgreSQL）进行 JOIN 操作

**示例对比**:
```sql
-- 标准 SQL（不支持）
SELECT p.*, c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE p.price > 100;

-- Milvus 替代方案（在应用层实现）
products = collection.query(expr="price > 100", ...)
category_ids = [p.category_id for p in products]
# 在应用层或另一个数据库查询 categories
```

### 2.2 复杂聚合函数

**不支持**: 不支持 `GROUP BY`、`SUM`、`AVG`、聚合 JOIN 等复杂聚合操作。

**限制**:
- Milvus 主要用于检索、过滤与向量搜索
- 不适合作大规模复杂算子计算
- 聚合操作需要在应用层实现

**示例对比**:
```sql
-- 标准 SQL（不支持）
SELECT category, COUNT(*), AVG(price)
FROM products
GROUP BY category;

-- Milvus 替代方案（在应用层实现）
results = collection.query(expr="", output_fields=["category", "price"], limit=10000)
# 在应用层进行聚合计算
```

### 2.3 事务和 ACID 操作

**不支持**: 虽然 Milvus 设计中有强一致性与最终一致性选项，也有一定的并发写入与读写隔离，但它**不完整支持事务操作**。

**限制**:
- 不像传统关系库那样支持完整的事务操作
- 不支持 `BEGIN TRANSACTION`、`COMMIT`、`ROLLBACK` 等
- 需要应用层处理一致性保证

### 2.4 完整的 SQL DDL 操作

**不支持**: Milvus 有自己的 Schema API 用于创建 collection、定义字段、索引类型等，**不一定支持标准 SQL 的 DDL 语句**。

**对比**:
```sql
-- 标准 SQL DDL（不支持）
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    price FLOAT
);

ALTER TABLE products ADD COLUMN category VARCHAR(100);
DROP INDEX idx_price;

-- Milvus Schema API（使用 SDK）
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="price", dtype=DataType.FLOAT),
]
schema = CollectionSchema(fields=fields)
collection = Collection("products", schema=schema)
```

### 2.5 排序（ORDER BY）

**限制**: 标量字段上的排序支持很有限，不像关系库那样可以在任意字段做排序 + 分页 + offset。

**特点**:
- 只能按向量相似度排序（`ORDER BY similarity`）
- 标量字段排序功能有限
- 通常以 `limit`/`top-k` 的方式通过向量 + filter + limit 获取结果

**示例对比**:
```sql
-- 标准 SQL（有限支持）
SELECT * FROM products
ORDER BY price DESC, created_at ASC
LIMIT 20 OFFSET 40;

-- Milvus（只能按向量相似度排序）
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2"},
    expr="",  # 可以在过滤条件中限制范围
    limit=20,  # 但不能实现真正的 offset
    output_fields=["id", "name", "price"]
)
```

---

## 三、SQL 与 Milvus 查询对比表

| 用例 | 标准 SQL | Milvus 实现方式 |
|------|---------|----------------|
| **查询某日期范围内特定类别的向量相似度前 10 个** | `SELECT * FROM products WHERE category = 'Electronics' AND date >= '2024-01-01' AND date <= '2024-12-31' ORDER BY similarity(vector, query_vector) LIMIT 10;` | `collection.search(data=[query_vector], anns_field="embedding", expr="category == 'Electronics' AND date >= '2024-01-01' AND date <= '2024-12-31'", limit=10)` |
| **获取主键为某些值的实体** | `SELECT * FROM users WHERE id IN (1,2,3);` | `collection.get(ids=[1, 2, 3], output_fields=[...])` |
| **分页查询某类实体** | `SELECT * FROM items WHERE color = 'blue' ORDER BY id LIMIT 20 OFFSET 40;` | `collection.query_iterator(expr="color == 'blue'", batch_size=20)` + 客户端实现分页 |
| **字符串匹配查询** | `SELECT * FROM products WHERE name LIKE 'iPhone%';` | `collection.query(expr="name like 'iPhone%'", ...)` |
| **数组字段查询** | `SELECT * FROM items WHERE 'popular' = ANY(tags);` | `collection.query(expr="array_contains(tags, 'popular')", ...)` |
| **多表 JOIN** | `SELECT p.*, c.name FROM products p JOIN categories c ON p.category_id = c.id;` | ❌ **不支持**，需要在应用层实现 |
| **聚合查询** | `SELECT category, COUNT(*), AVG(price) FROM products GROUP BY category;` | ❌ **不支持**，需要在应用层实现 |
| **事务操作** | `BEGIN; INSERT ...; UPDATE ...; COMMIT;` | ❌ **不支持**，Milvus 有最终一致性但无完整事务 |

---

## 四、Milvus 查询表达式语法

### 4.1 基本语法

```python
# 比较运算符
expr = "price > 100"
expr = "price >= 100 AND price <= 1000"
expr = "category == 'electronics'"
expr = "status != 'deleted'"

# 字符串匹配
expr = "name like 'iPhone%'"  # 前缀匹配
expr = "name like '%phone%'"  # 包含匹配
expr = "name like '%phone'"   # 后缀匹配

# 布尔组合
expr = "category == 'electronics' AND price > 100"
expr = "category == 'electronics' OR category == 'computers'"
expr = "NOT status == 'deleted'"

# 数组操作
expr = "array_contains(tags, 'popular')"
expr = "array_length(tags) > 0"

# JSON 字段操作
expr = "metadata['author'] == 'John'"
expr = "metadata['price'] > 100"
```

### 4.2 数据类型支持

**支持的标量类型**:
- **整数**: `INT8`, `INT16`, `INT32`, `INT64`
- **浮点数**: `FLOAT`, `DOUBLE`
- **字符串**: `VARCHAR`, `STRING`
- **布尔**: `BOOL`
- **数组**: `ARRAY<INT64>`, `ARRAY<VARCHAR>` 等
- **JSON**: `JSON`

**向量类型**:
- **浮点向量**: `FLOAT_VECTOR`
- **二进制向量**: `BINARY_VECTOR`

---

## 五、完整查询示例

### 5.1 创建 Collection

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections

# 连接到 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="price", dtype=DataType.FLOAT),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
]
schema = CollectionSchema(fields=fields, description="Product collection")

# 创建 Collection
collection = Collection("products", schema=schema)

# 创建索引
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)
```

### 5.2 插入数据

```python
import random

# 准备数据
data = [
    [i for i in range(100)],  # ids
    [f"product_{i}" for i in range(100)],  # names
    [random.choice(["electronics", "computers", "phones"]) for _ in range(100)],  # categories
    [random.uniform(10, 1000) for _ in range(100)],  # prices
    [[random.random() for _ in range(128)] for _ in range(100)],  # embeddings
]

# 插入数据
collection.insert(data)
collection.flush()  # 确保数据持久化
```

### 5.3 执行查询

```python
# 1. 基础过滤查询
results = collection.query(
    expr="category == 'electronics' AND price > 100",
    output_fields=["id", "name", "category", "price"],
    limit=10
)

# 2. 向量相似度搜索 + 过滤
query_vector = [[random.random() for _ in range(128)]]
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    expr="category == 'electronics' AND price > 100",
    limit=10,
    output_fields=["id", "name", "category", "price"]
)

# 3. 按主键查询
results = collection.get(
    ids=[1, 2, 3, 4, 5],
    output_fields=["id", "name", "category", "price"]
)

# 4. 分页查询
query_iterator = collection.query_iterator(
    expr="category == 'electronics'",
    output_fields=["id", "name", "category", "price"],
    batch_size=20
)

while True:
    batch = query_iterator.next()
    if not batch:
        break
    # 处理批次数据
    for item in batch:
        print(item)
```

---

## 六、Milvus vs 标准 SQL 对比总结

### 6.1 支持的能力

| 能力 | Milvus | 说明 |
|------|--------|------|
| **过滤表达式** | ✅ 支持 | 支持比较、布尔、字符串匹配、数组操作 |
| **按主键查询** | ✅ 支持 | `Get` 操作 |
| **向量相似度搜索** | ✅ 核心功能 | 专为向量搜索优化 |
| **混合搜索** | ✅ 支持 | 向量搜索 + 过滤条件 |
| **分页查询** | ⚠️ 有限支持 | 通过 `QueryIterator`，但有范围限制 |

### 6.2 不支持的能力

| 能力 | Milvus | 替代方案 |
|------|--------|---------|
| **JOIN** | ❌ 不支持 | 应用层实现或使用其他数据库 |
| **复杂聚合** | ❌ 不支持 | 应用层实现聚合计算 |
| **完整事务** | ❌ 不支持 | 使用最终一致性模型 |
| **标准 DDL** | ❌ 不支持 | 使用 Milvus Schema API |
| **任意字段排序** | ❌ 不支持 | 只能按向量相似度排序 |

---

## 七、使用建议

### 7.1 何时使用 Milvus

**适合的场景**:
- 大规模向量相似度搜索
- 需要结合 metadata 过滤的向量搜索
- 实时推荐系统
- 图像/视频检索
- 语义搜索

### 7.2 何时不使用 Milvus

**不适合的场景**:
- 需要复杂 JOIN 操作
- 需要复杂聚合计算
- 需要完整事务支持
- 需要标准 SQL 接口

**替代方案**:
- 使用 **pgvector**（PostgreSQL 扩展）：提供标准 SQL 接口 + 向量搜索
- 使用 **Qdrant**：提供 REST/gRPC API，支持丰富的过滤表达式
- 混合架构：Milvus 负责向量搜索，其他数据库负责复杂 SQL 操作

---

## 八、与其他向量数据库对比

### 8.1 Milvus vs pgvector

| 特性 | Milvus | pgvector |
|------|--------|----------|
| **SQL 支持** | 类 SQL（有限） | ✅ 完整 SQL 支持 |
| **向量搜索性能** | ✅ 高性能 | ✅ 高性能 |
| **扩展性** | ✅ 分布式 | ⚠️ 单机/主从 |
| **事务支持** | ❌ 不支持 | ✅ 完整 ACID |
| **学习曲线** | 中等（需要学习 SDK） | 低（标准 SQL） |

### 8.2 Milvus vs Qdrant

| 特性 | Milvus | Qdrant |
|------|--------|--------|
| **SQL 支持** | 类 SQL（有限） | ❌ 不支持（但有 REST/gRPC API） |
| **过滤表达式** | ✅ 支持 | ✅ 支持（40+ 算子） |
| **向量搜索性能** | ✅ 高性能 | ✅ 高性能 |
| **扩展性** | ✅ 分布式 | ✅ 分布式 |
| **API 接口** | Python/Go/Java SDK | REST/gRPC API |

---

## 九、总结

### 9.1 核心结论

1. **Milvus 不直接支持标准 SQL**，但提供了**类 SQL 的查询能力**
2. **主要支持**：
   - 过滤表达式（metadata/scalar 字段）
   - 向量相似度搜索
   - 混合搜索（向量 + 过滤）
   - 按主键查询
3. **不支持**：
   - JOIN 操作
   - 复杂聚合函数
   - 完整事务支持
   - 标准 SQL DDL

### 9.2 使用建议

- **如果主要需要向量搜索 + 简单过滤**：Milvus 是很好的选择
- **如果需要复杂 SQL 操作**：考虑使用 pgvector 或混合架构
- **如果需要标准 SQL 接口**：推荐使用 pgvector（PostgreSQL 扩展）

### 9.3 未来展望

- Milvus 可能会继续增强类 SQL 能力
- 但不太可能完全支持标准 SQL（因为架构设计重点在向量搜索）
- 建议关注官方文档了解最新特性

---

## 参考资料

- [Milvus 官方文档 - 查询表达式](https://milvus.io/docs/get-and-scalar-query.md)
- [Milvus 官方文档 - 过滤表达式](https://milvus.io/docs/generating_milvus_query_filter_expressions.md)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
- [Milvus vs pgvector 对比](https://milvus.io/docs/v2.4.x/comparison.md)
