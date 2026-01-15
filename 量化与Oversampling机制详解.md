# 量化与 Oversampling 机制详解

## 概述

**Oversampling（过采样）** 是 Qdrant 在使用量化向量进行搜索时的一种准确性补偿机制。由于量化会引入误差，导致基于量化向量的排序可能不准确，因此需要先搜索更多的候选结果，然后用原始向量重新评分，最终返回准确的 Top-K。

---

## 一、量化的基本原理

### 1.1 什么是量化

**量化（Quantization）** 是一种向量压缩技术，通过降低向量的精度来减少存储空间和计算量。

#### 1.1.1 量化类型

Qdrant 支持多种量化类型：

| 量化类型 | 说明 | 压缩比 | 精度损失 |
|---------|------|--------|---------|
| **Scalar Quantization (SQ)** | 标量量化，将 float32 压缩为 uint8 | ~4x | 低 |
| **Product Quantization (PQ)** | 乘积量化，将向量分成多个子向量分别量化 | ~8-16x | 中 |
| **Binary Quantization** | 二值量化，将向量压缩为二进制 | ~32x | 高 |

#### 1.1.2 量化示例

**原始向量**（float32，128 维）：
```
[0.123, 0.456, 0.789, ...]  →  512 字节
```

**量化后**（uint8，128 维）：
```
[31, 116, 201, ...]  →  128 字节（压缩 4 倍）
```

### 1.2 量化的优势

1. **存储空间减少**：压缩比可达 4-32 倍
2. **计算速度提升**：量化向量计算更快
3. **内存占用降低**：可以加载更多向量到内存

### 1.3 量化的代价：精度损失

量化会引入**误差**，导致：

1. **距离计算不准确**：量化向量的距离 ≠ 原始向量的距离
2. **排序可能错误**：基于量化向量的排序可能与原始向量不同
3. **Top-K 可能遗漏**：真正的 Top-K 可能不在量化搜索的前 K 个结果中

---

## 二、量化误差导致的排序问题

### 2.1 问题场景

假设我们有一个查询向量 `q` 和两个候选向量 `a` 和 `b`：

```
原始向量：
  distance(q, a) = 0.10  ← 更相似
  distance(q, b) = 0.15  ← 较不相似

量化向量：
  distance(q_quantized, a_quantized) = 0.12
  distance(q_quantized, b_quantized) = 0.11  ← 排序错误！
```

**问题**：使用量化向量时，`b` 被排在了 `a` 前面，但实际 `a` 更相似。

### 2.2 误差累积

在 HNSW 图搜索中，误差会累积：

```
1. 图遍历时使用量化向量计算距离
2. 选择下一个节点时可能选错
3. 最终 Top-K 结果可能包含错误的排序
```

### 2.3 实际影响

**场景**：用户请求 Top-10

```
使用量化向量搜索：
  返回：10 个结果
  问题：可能遗漏真正的 Top-10 中的某些结果

原因：
  - 量化向量的排序不准确
  - 真正的 Top-10 可能分散在量化搜索的前 20、30 甚至更多结果中
```

---

## 三、Oversampling 的解决方案

### 3.1 核心思想

**Oversampling** 通过以下步骤解决量化误差问题：

```
1. 用量化向量搜索更多候选（oversampling × limit）
2. 用原始向量重新评分（rescoring）
3. 重新排序并取 Top-K
```

### 3.2 完整流程

```
用户请求：Top-10，oversampling = 2.4

步骤 1：量化向量搜索
  └─ 搜索 24 个候选结果（10 × 2.4 = 24）
  └─ 使用量化向量计算距离（快速但可能有误差）

步骤 2：原始向量重新评分（Rescoring）
  └─ 对 24 个候选结果，使用原始向量重新计算距离
  └─ 得到准确的相似度分数

步骤 3：重新排序
  └─ 按原始向量的分数重新排序
  └─ 取 Top-10

步骤 4：返回结果
  └─ 返回准确的 Top-10
```

### 3.3 代码实现

#### 3.3.1 Oversampling 计算

**位置**: `lib/segment/src/index/vector_index_search_common.rs:27-45`

```rust
pub fn get_oversampled_top(
    quantized_storage: Option<&QuantizedVectors>,
    params: Option<&SearchParams>,
    top: usize,
) -> usize {
    let quantization_enabled = is_quantized_search(quantized_storage, params);

    let oversampling_value = params
        .and_then(|p| p.quantization)
        .map(|q| q.oversampling)
        .unwrap_or(default_quantization_oversampling_value());

    match oversampling_value {
        Some(oversampling) if quantization_enabled && oversampling > 1.0 => {
            (oversampling * top as f64) as usize  // 例如：2.4 * 10 = 24
        }
        _ => top,  // 如果没有量化或 oversampling = 1.0，返回原始 top
    }
}
```

#### 3.3.2 HNSW 搜索中使用 Oversampling

**位置**: `lib/segment/src/index/hnsw_index/hnsw.rs:1026-1124`

```rust
fn search_with_graph(
    &self,
    vector: &QueryVector,
    filter: Option<&Filter>,
    top: usize,  // 用户请求的 top（如 10）
    params: Option<&SearchParams>,
    // ...
) -> OperationResult<Vec<ScoredPointOffset>> {
    // ========== 步骤 1：计算 oversampled_top ==========
    let oversampled_top = get_oversampled_top(quantized_vectors.as_ref(), params, top);
    // 例如：top=10, oversampling=2.4 → oversampled_top=24

    // ========== 步骤 2：使用量化向量搜索更多候选 ==========
    let points_scorer = Self::construct_search_scorer(
        vector,
        &vector_storage,
        quantized_vectors.as_ref(),  // 使用量化向量
        deleted_points,
        params,
        vector_query_context.hardware_counter(),
        filter_context,
    )?;

    // 搜索 oversampled_top 个结果（而不是 top 个）
    let search_result = self.graph.search(
        oversampled_top,  // 24 个候选
        ef,
        algorithm,
        points_scorer,  // 使用量化向量计算距离
        custom_entry_points,
        &is_stopped,
    )?;

    // ========== 步骤 3：后处理（包括 rescoring） ==========
    postprocess_search_result(
        search_result,  // 24 个候选结果
        id_tracker.deleted_point_bitslice(),
        &vector_storage,
        quantized_vectors.as_ref(),
        vector,
        params,
        top,  // 最终返回 10 个
        vector_query_context.hardware_counter(),
    )
}
```

#### 3.3.3 Rescoring 实现

**位置**: `lib/segment/src/index/vector_index_search_common.rs:48-87`

```rust
pub fn postprocess_search_result(
    mut search_result: Vec<ScoredPointOffset>,  // 24 个候选结果
    point_deleted: &BitSlice,
    vector_storage: &VectorStorageEnum,
    quantized_vectors: Option<&QuantizedVectors>,
    vector: &QueryVector,
    params: Option<&SearchParams>,
    top: usize,  // 10
    hardware_counter: HardwareCounterCell,
) -> OperationResult<Vec<ScoredPointOffset>> {
    let quantization_enabled = is_quantized_search(quantized_vectors, params);

    // ========== 检查是否需要 rescoring ==========
    let default_rescoring = quantized_vectors
        .as_ref()
        .map(|q| q.default_rescoring())
        .unwrap_or(false);
    let rescore = quantization_enabled
        && params
            .and_then(|p| p.quantization)
            .and_then(|q| q.rescore)
            .unwrap_or(default_rescoring);

    // ========== 如果需要，使用原始向量重新评分 ==========
    if rescore {
        // 创建使用原始向量的 scorer（不使用量化向量）
        let mut scorer = FilteredScorer::new(
            vector.to_owned(),
            vector_storage,  // 原始向量存储
            None,  // 不使用量化向量
            None,
            point_deleted,
            hardware_counter,
        )?;

        // 对 24 个候选结果重新评分
        search_result = scorer
            .score_points(&mut search_result.iter().map(|x| x.idx).collect_vec(), 0)
            .collect();
        
        // 重新排序
        search_result.sort_unstable();
        search_result.reverse();
    }

    // ========== 截取 Top-K ==========
    search_result.truncate(top);  // 从 24 个中取 Top-10
    Ok(search_result)
}
```

---

## 四、为什么需要 Oversampling

### 4.1 量化误差的分布

量化误差不是均匀分布的，而是**集中在某些区域**：

```
高相似度区域：
  - 误差相对较小
  - 排序相对准确

中等相似度区域：
  - 误差较大
  - 排序可能错误

低相似度区域：
  - 误差很大
  - 排序基本不可靠
```

### 4.2 排序错误的概率

**问题**：如果只搜索 `top` 个结果，可能遗漏真正的 Top-K。

**示例**：

```
真正的 Top-10：
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

量化搜索的前 10 个：
  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  ↑ 遗漏了 2, 4, 6, 8, 10

量化搜索的前 24 个（oversampling = 2.4）：
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  ↑ 包含了所有真正的 Top-10
```

### 4.3 Oversampling 的作用

**Oversampling 通过扩大候选集，增加找到真正 Top-K 的概率**：

| Oversampling | 候选数量 | 找到真正 Top-10 的概率 |
|-------------|---------|---------------------|
| 1.0 | 10 | ~60-70% |
| 2.0 | 20 | ~90-95% |
| 2.4 | 24 | ~95-98% |
| 4.0 | 40 | ~99%+ |

---

## 五、Oversampling 参数配置

### 5.1 参数定义

**位置**: `lib/segment/src/types.rs:495-505`

```rust
/// Oversampling factor for quantization. Default is 1.0.
///
/// Defines how many extra vectors should be pre-selected using quantized index,
/// and then re-scored using original vectors.
///
/// For example, if `oversampling` is 2.4 and `limit` is 100, then 240 vectors will be pre-selected using quantized index,
/// and then top-100 will be returned after re-scoring.
pub oversampling: Option<f64>,
```

### 5.2 默认值

- **默认值**：`None`（相当于 1.0，不进行 oversampling）
- **最小值**：1.0
- **推荐值**：2.0 - 4.0（根据量化类型和精度要求）

### 5.3 不同量化类型的默认 Rescoring

**位置**: `lib/segment/src/vector_storage/quantized/quantized_vectors.rs:194-215`

```rust
pub fn default_rescoring(&self) -> bool {
    match self.storage_impl {
        // Scalar 和 PQ：默认不 rescore（精度损失较小）
        QuantizedVectorStorage::ScalarRam(_) => false,
        QuantizedVectorStorage::PQRam(_) => false,
        // Binary：默认 rescore（精度损失较大）
        QuantizedVectorStorage::BinaryRam(_) => true,
        // ...
    }
}
```

**说明**：
- **Scalar/PQ 量化**：精度损失较小，默认不 rescore
- **Binary 量化**：精度损失较大，默认 rescore

---

## 六、完整示例

### 6.1 场景设置

```
用户请求：
  - limit: 100
  - 量化类型：Product Quantization (PQ)
  - oversampling: 2.4
  - rescore: true（显式启用）
```

### 6.2 执行流程

```
步骤 1：计算 oversampled_top
  oversampled_top = 100 × 2.4 = 240

步骤 2：HNSW 搜索（使用量化向量）
  输入：
    - 查询向量：q
    - top: 240
    - 使用：PQ 量化向量
  输出：
    - 240 个候选结果（基于量化向量的分数排序）

步骤 3：Rescoring（使用原始向量）
  输入：
    - 240 个候选结果
    - 原始向量存储
  过程：
    - 对每个候选结果，使用原始向量重新计算距离
    - 得到准确的相似度分数
  输出：
    - 240 个结果（基于原始向量的分数排序）

步骤 4：截取 Top-K
  输入：240 个重新排序的结果
  输出：Top-100

步骤 5：返回给用户
  └─ 100 个准确的结果
```

### 6.3 性能对比

| 方法 | 搜索数量 | Rescoring 数量 | 准确性 | 性能 |
|------|---------|---------------|--------|------|
| **无量化** | 100 | 0 | 100% | 慢 |
| **量化 + 无 Oversampling** | 100 | 100 | ~70% | 快 |
| **量化 + Oversampling 2.4** | 240 | 240 | ~98% | 中等 |

---

## 七、Oversampling 与 Undersampling 的区别

### 7.1 对比表

| 特性 | Oversampling | Undersampling |
|------|-------------|---------------|
| **目的** | 补偿量化误差，提高准确性 | 减少网络传输，提高性能 |
| **适用场景** | 量化搜索 | 分布式搜索（大 limit） |
| **方向** | 搜索更多结果（扩大候选集） | 搜索更少结果（减少传输） |
| **数学基础** | 量化误差分析 | Poisson 分布 |
| **处理位置** | Segment 级别 | Collection/Shard 级别 |

### 7.2 可以同时使用

**场景**：分布式环境 + 量化向量

```
用户请求：Top-1000，10 个分片，量化 + oversampling=2.4

Undersampling（分片级别）：
  - 每个分片返回：~171 个结果（而不是 1000 个）

Oversampling（Segment 级别）：
  - 每个 Segment 搜索：171 × 2.4 = 410 个候选
  - Rescoring 后返回：171 个结果

最终：
  - 每个分片返回：171 个结果
  - 合并后取 Top-1000
```

---

## 八、代码位置总结

### 8.1 核心文件

| 文件 | 功能 |
|------|------|
| `lib/segment/src/index/vector_index_search_common.rs` | Oversampling 计算和 Rescoring |
| `lib/segment/src/index/hnsw_index/hnsw.rs` | HNSW 搜索中使用 Oversampling |
| `lib/segment/src/types.rs` | QuantizationSearchParams 定义 |

### 8.2 关键函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `get_oversampled_top` | `vector_index_search_common.rs:27` | 计算 oversampled_top |
| `postprocess_search_result` | `vector_index_search_common.rs:48` | Rescoring 和后处理 |
| `search_with_graph` | `hnsw_index/hnsw.rs:994` | HNSW 搜索主函数 |

### 8.3 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `oversampling` | `Option<f64>` | `None` (1.0) | Oversampling 因子 |
| `rescore` | `Option<bool>` | 根据量化类型 | 是否重新评分 |

---

## 九、总结

### 9.1 核心原理

**Oversampling 解决量化误差问题**：
1. 量化会引入误差，导致排序不准确
2. 通过搜索更多候选（oversampling × limit），增加找到真正 Top-K 的概率
3. 使用原始向量重新评分（rescoring），得到准确的排序
4. 最终返回准确的 Top-K

### 9.2 关键优势

1. **准确性提升**：从 ~70% 提升到 ~98%
2. **性能平衡**：在准确性和性能之间取得平衡
3. **灵活配置**：可以根据量化类型和精度要求调整

### 9.3 适用场景

- ✅ 使用量化向量搜索
- ✅ 需要高准确性的场景
- ✅ 可以接受一定的性能开销（rescoring）
- ❌ 精确搜索（会禁用量化）

### 9.4 设计权衡

Qdrant 选择了**准确性优先**的策略：
- 接受一定的性能开销（搜索更多候选 + rescoring）
- 换取更高的准确性（从 ~70% 提升到 ~98%）

这种权衡在大多数场景下是合理的，因为：
1. 量化已经带来了显著的性能提升
2. 准确性是搜索系统的核心要求
3. Oversampling 的开销相对可控

---

## 参考资料

- [Qdrant 中 Undersampling 算法逻辑详解](./Qdrant中Undersampling算法逻辑详解.md)
- [分布式环境下 HNSW 向量搜索工作原理](./分布式环境下HNSW向量搜索工作原理.md)
- [Product Quantization - Wikipedia](https://en.wikipedia.org/wiki/Product_quantization)
