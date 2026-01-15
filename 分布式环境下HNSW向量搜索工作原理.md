# 分布式环境下基于 HNSW 的近似向量搜索工作原理

## 概述

在分布式环境下，Qdrant 的向量搜索确实是一个 **Top-N 问题**。每个分片（Shard）的 HNSW 索引返回一定数量的候选结果，然后在协调节点（Coordinator）进行合并（Merge），最终返回全局的 Top-N 结果。

---

## 一、核心问题：Top-N 的分布式实现

### 1.1 问题的本质

**用户的问题**：在分布式环境下，基于 HNSW 的近似向量搜索是如何工作的？是否每个 HNSW 索引返回 n 个向量，最后做一次 merge？

**答案**：**是的，但更复杂一些**。

### 1.2 基本流程

```
用户请求 Top-10
    ↓
协调节点分发请求到所有分片
    ↓
每个分片：HNSW 搜索返回候选结果（可能 > 10）
    ↓
协调节点：合并所有分片的结果
    ↓
协调节点：取全局 Top-10
    ↓
返回给用户
```

---

## 二、详细工作流程

### 2.1 架构层次

Qdrant 的分布式搜索涉及多个层次：

```
Collection（集合）
    ↓
Shard（分片）- 可能在不同节点
    ↓
Segment（段）- 在同一个分片内
    ↓
HNSW Index（HNSW 索引）- 在每个段内
```

### 2.2 搜索流程

#### 步骤 1：用户请求

```rust
// 用户请求：Top-10 相似向量
search_request = {
    query_vector: [0.1, 0.2, ...],
    limit: 10,  // 返回 10 个结果
    offset: 0,
    // ...
}
```

#### 步骤 2：协调节点分发请求

**位置**: `lib/collection/src/collection/search.rs:146-200`

```rust
async fn do_core_search_batch(
    &self,
    request: CoreSearchRequestBatch,
    // ...
) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
    // 1. 选择目标分片
    let shard_holder = self.shards_holder.read().await;
    let target_shards = shard_holder.select_shards(shard_selection)?;
    
    // 2. 并发查询所有分片
    let all_searches = target_shards.into_iter().map(|(shard, shard_key)| {
        shard.core_search(
            request.clone(),
            read_consistency,
            shard_selection.is_shard_id(),
            timeout,
            hw_measurement_acc.clone(),
        )
    });
    
    let all_searches_res = future::try_join_all(all_searches).await?;
    
    // 3. 合并结果
    let result = self.merge_from_shards(
        all_searches_res,
        request.clone(),
        !shard_selection.is_shard_id(),
    ).await;
    
    result
}
```

**关键点**：
- **并发查询**：所有分片同时进行搜索
- **每个分片返回**：`limit` 个结果（或经过优化后的数量）

#### 步骤 3：分片内部搜索

**位置**: `lib/collection/src/shards/local_shard/search.rs`

每个分片内部可能有多个 Segment，每个 Segment 都有自己的 HNSW 索引：

```rust
// 分片内部搜索
// 1. 在每个 Segment 上搜索
// 2. 合并 Segment 的结果
// 3. 返回 Top-K 给协调节点
```

#### 步骤 4：Segment 级别的 HNSW 搜索

**位置**: `lib/segment/src/index/hnsw_index/hnsw.rs:994-1145`

```rust
fn search_with_graph(
    &self,
    vector: &QueryVector,
    filter: Option<&Filter>,
    top: usize,  // 请求的 Top-K
    params: Option<&SearchParams>,
    // ...
) -> OperationResult<Vec<ScoredPointOffset>> {
    // 1. 获取 ef 参数（候选列表大小）
    let ef = params
        .and_then(|params| params.hnsw_ef)
        .unwrap_or(self.config.ef);  // 默认使用配置的 ef
    
    // 2. 计算 oversampled_top（可能 > top）
    let oversampled_top = get_oversampled_top(quantized_vectors.as_ref(), params, top);
    
    // 3. HNSW 搜索：维护 ef 个候选，返回 top 个结果
    let search_result = self.graph.search(
        oversampled_top,  // 可能 > top
        ef,               // 候选列表大小
        algorithm,
        points_scorer,
        custom_entry_points,
        &is_stopped,
    )?;
    
    // 4. 后处理：最终返回 top 个结果
    postprocess_search_result(
        search_result,
        // ...
        top,  // 最终返回 top 个
    )
}
```

**关键参数**：

1. **`ef`（ef_search）**：
   - HNSW 搜索时维护的候选列表大小
   - 通常 `ef >= top`
   - 例如：`ef = 100`，`top = 10` → 搜索时维护 100 个候选，返回 10 个

2. **`top`**：
   - 最终返回的结果数量
   - Segment 级别：返回 `top` 个结果

3. **`oversampled_top`**：
   - 如果使用量化（Quantization），可能会过采样
   - 例如：`top = 10`，`oversampling = 1.5` → `oversampled_top = 15`

**位置**: `lib/segment/src/index/hnsw_index/graph_layers.rs:530-561`

```rust
pub fn search(
    &self,
    top: usize,      // 返回 top 个结果
    ef: usize,      // 搜索时维护 ef 个候选
    algorithm: SearchAlgorithm,
    mut points_scorer: FilteredScorer,
    // ...
) -> CancellableResult<Vec<ScoredPointOffset>> {
    // ...
    let ef = max(ef, top);  // ef 至少等于 top
    
    // 在 HNSW 图中搜索，维护 ef 个候选
    let nearest = match algorithm {
        SearchAlgorithm::Hnsw => {
            self.search_on_level(zero_level_entry, 0, ef, &mut points_scorer, is_stopped)
        }
        // ...
    }?;
    
    // 返回 top 个结果
    Ok(nearest.into_iter_sorted().take(top).collect_vec())
}
```

#### 步骤 5：协调节点合并结果

**位置**: `lib/collection/src/collection/search.rs:262-314`

```rust
async fn merge_from_shards(
    &self,
    mut all_searches_res: Vec<Vec<Vec<ScoredPoint>>>,  // [shard][batch][point]
    request: Arc<CoreSearchRequestBatch>,
    is_client_request: bool,
) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
    // ...
    
    for (batch_index, request) in request.searches.iter().enumerate() {
        // 1. 确定排序顺序（距离越小越好，还是分数越大越好）
        let order = if request.query.is_distance_scored() {
            collection_params
                .get_distance(request.query.get_vector_name())?
                .distance_order()
        } else {
            Order::LargeBetter
        };
        
        // 2. 从所有分片收集结果
        let results_from_shards = all_searches_res
            .iter_mut()
            .map(|res| res.get_mut(batch_index).map_or(Vec::new(), mem::take));
        
        // 3. 使用 k-way merge 合并结果
        let merged_iter = match order {
            Order::LargeBetter => Either::Left(
                results_from_shards.kmerge_by(|a, b| a > b)  // 分数大的优先
            ),
            Order::SmallBetter => Either::Right(
                results_from_shards.kmerge_by(|a, b| a < b)  // 距离小的优先
            ),
        }
        .filter(|point| seen_ids.insert(point.id));  // 去重
        
        // 4. 取 Top-K（考虑 offset 和 limit）
        let top_res = if is_client_request && request.offset > 0 {
            merged_iter
                .skip(request.offset)
                .take(request.limit)
                .collect()
        } else {
            merged_iter.take(request.offset + request.limit).collect()
        };
        
        top_results.push(top_res);
    }
    
    Ok(top_results)
}
```

**关键点**：

1. **K-way Merge**：
   - 使用 `kmerge_by` 进行多路归并
   - 按分数/距离排序
   - 流式处理，不需要加载所有结果到内存

2. **去重**：
   - 使用 `seen_ids` 集合去重
   - 防止同一个点在不同分片出现

3. **Top-K 选择**：
   - 合并后取前 `limit` 个
   - 支持 `offset`（分页）

---

## 三、每个分片返回多少结果？

### 3.1 基本规则

**每个分片返回 `limit` 个结果**（或经过优化后的数量）。

### 3.2 特殊情况：Undersampling 优化

**位置**: `lib/collection/src/collection/query.rs:91-144`

当 `limit` 很大时，Qdrant 会进行 **Undersampling 优化**：

```rust
/// If the query limit above this value, it will be a subject to undersampling.
const SHARD_QUERY_SUBSAMPLING_LIMIT: usize = 128;

/// Give some more ensurance for undersampling,
/// retrieve more points to prevent undersampling errors.
const MORE_ENSURANCE_FACTOR: f64 = 1.2;

fn modify_shard_query_for_undersampling_limits(
    batch_request: Arc<Vec<ShardQueryRequest>>,
    num_shards: usize,
    is_auto_sharding: bool,
) -> Arc<Vec<ShardQueryRequest>> {
    // ...
    
    // 如果 limit >= 128 且是自动分片
    if max_limit >= Self::SHARD_QUERY_SUBSAMPLING_LIMIT && is_auto_sharding {
        // 计算每个分片的 limit
        // 例如：1000 limit, 10 shards
        // lambda = 1.0 / 10 * 1.2 * 1000 = 120
        // 每个分片查询 171 个结果（而不是 1000 个）
        let undersample_limit = find_search_sampling_over_point_distribution(
            request_limit as f64,
            1. / num_shards as f64 * Self::MORE_ENSURANCE_FACTOR,
        );
        
        new_request.limit = std::cmp::min(undersample_limit, request_limit);
    }
    
    // ...
}
```

**优化原理**：

- **问题**：如果用户请求 Top-1000，每个分片都返回 1000 个结果，网络传输和内存占用都很大
- **解决方案**：利用数据随机分布的特性，每个分片只返回更少的结果（如 171 个），然后合并
- **风险**：可能错过一些真正的 Top-K，但概率很低（因为数据随机分布）

**示例**：

```
用户请求：Top-1000
分片数量：10
优化后：每个分片返回 ~171 个结果（而不是 1000 个）
合并后：从 10 * 171 = 1710 个候选中选择 Top-1000
```

### 3.3 Segment 级别的结果数量

**位置**: `lib/segment/src/index/hnsw_index/hnsw.rs:1026`

```rust
// Segment 级别可能返回更多结果（oversampling）
let oversampled_top = get_oversampled_top(quantized_vectors.as_ref(), params, top);
```

**Oversampling**：

- 如果使用量化（Product Quantization），可能会过采样
- 例如：请求 Top-10，可能返回 15 个候选，然后重新评分，最终返回 Top-10

---

## 四、HNSW 搜索的 ef 参数

### 4.1 ef 的作用

**`ef`（ef_search）** 是 HNSW 搜索的关键参数：

- **定义**：搜索时维护的候选列表大小
- **作用**：控制搜索的准确性和速度
- **关系**：`ef >= top`（通常 `ef = 2 * top` 到 `10 * top`）

### 4.2 ef 在分布式搜索中的影响

**每个 Segment 的 HNSW 搜索**：

```rust
// Segment 级别
let ef = params.hnsw_ef.unwrap_or(self.config.ef);  // 例如：ef = 100
let top = request.limit;  // 例如：top = 10

// HNSW 搜索：维护 100 个候选，返回 10 个结果
self.graph.search(top, ef, ...)
```

**关键点**：

1. **`ef` 只影响 Segment 级别的搜索质量**：
   - 更大的 `ef` → 更准确的 Segment 级别结果
   - 但不影响分片返回的结果数量

2. **分片返回的结果数量 = `limit`**（或优化后的数量）

3. **合并后的结果数量 = `limit`**

---

## 五、完整示例

### 5.1 场景设置

```
- 用户请求：Top-10
- 分片数量：3
- 每个分片的 Segment 数量：2
- HNSW ef 参数：50
```

### 5.2 搜索流程

```
1. 用户请求 Top-10
   ↓
2. 协调节点分发到 3 个分片
   ↓
3. 每个分片内部：
   - 分片 1：
     * Segment 1: HNSW 搜索（ef=50）→ 返回 10 个候选
     * Segment 2: HNSW 搜索（ef=50）→ 返回 10 个候选
     * 合并 → 返回 Top-10 给协调节点
   - 分片 2：返回 Top-10
   - 分片 3：返回 Top-10
   ↓
4. 协调节点收到：
   - 分片 1: 10 个结果
   - 分片 2: 10 个结果
   - 分片 3: 10 个结果
   总共：30 个候选结果
   ↓
5. 协调节点 K-way Merge：
   - 按分数排序
   - 去重
   - 取 Top-10
   ↓
6. 返回给用户：Top-10
```

### 5.3 代码流程

```rust
// 1. 用户请求
search_request = { limit: 10, ... }

// 2. 协调节点分发
for shard in shards {
    shard_results = shard.core_search(search_request)  // 每个分片返回 10 个
}

// 3. 分片内部（每个分片）
for segment in segments {
    segment_results = segment.hnsw_search(
        top: 10,
        ef: 50,  // 维护 50 个候选，返回 10 个
    )
}
shard_results = merge_segments(segment_results)  // 合并后返回 10 个

// 4. 协调节点合并
all_results = [shard1_results, shard2_results, shard3_results]  // 30 个候选
final_results = kmerge_by(all_results).take(10)  // 取 Top-10
```

---

## 六、性能优化

### 6.1 并发搜索

**所有分片并发搜索**：

```rust
// 并发查询所有分片
let all_searches = target_shards.into_iter().map(|(shard, _)| {
    shard.core_search(...)  // 异步并发
});
let all_searches_res = future::try_join_all(all_searches).await?;
```

**优势**：
- 总搜索时间 ≈ 最慢分片的搜索时间
- 而不是所有分片搜索时间的总和

### 6.2 K-way Merge 优化

**流式合并**：

```rust
// 使用迭代器，不需要加载所有结果到内存
let merged_iter = results_from_shards.kmerge_by(|a, b| a > b);
let top_res = merged_iter.take(limit).collect();
```

**优势**：
- 内存效率高
- 只需要维护一个大小为 `limit` 的堆

### 6.3 Undersampling 优化

**减少网络传输**：

- 当 `limit` 很大时，每个分片返回更少的结果
- 减少网络传输和内存占用
- 利用数据随机分布的特性，保证准确性

---

## 七、关键参数总结

### 7.1 用户请求参数

- **`limit`**：最终返回的结果数量（Top-K）
- **`offset`**：分页偏移量

### 7.2 HNSW 搜索参数

- **`ef`（ef_search）**：
  - Segment 级别：搜索时维护的候选列表大小
  - 通常 `ef >= limit`
  - 默认值：通常等于 `ef_construct`

- **`ef_construct`**：
  - 构建索引时的候选数量
  - 影响索引质量

### 7.3 分布式搜索参数

- **分片返回数量**：
  - 正常情况下：`limit`
  - Undersampling 优化：`min(undersample_limit, limit)`

- **最终返回数量**：`limit`

---

## 八、与精确搜索的对比

### 8.1 精确搜索（Exact Search）

如果设置 `exact: true`：

```rust
let exact = params.map(|params| params.exact).unwrap_or(false);
if exact {
    // 使用暴力搜索（全扫描）
    // 每个 Segment 扫描所有向量
    // 返回 Top-K
}
```

**特点**：
- 每个 Segment 扫描所有向量
- 返回真正的 Top-K
- 但速度慢

### 8.2 近似搜索（Approximate Search）

使用 HNSW 索引：

```rust
// 使用 HNSW 索引
// 每个 Segment：维护 ef 个候选，返回 top 个
// 合并后：取全局 Top-K
```

**特点**：
- 速度快（对数复杂度）
- 可能不是真正的 Top-K（近似）
- 但召回率通常很高（>95%）

---

## 九、总结

### 9.1 核心答案

**用户的问题**：是否每个 HNSW 索引返回 n 个向量，最后做一次 merge？

**答案**：

1. **是的**，基本思路正确
2. **但更复杂**：
   - 每个 **Segment** 的 HNSW 索引返回 `top` 个结果（可能经过 oversampling）
   - 每个 **分片** 合并其内部所有 Segment 的结果，返回 `limit` 个结果
   - **协调节点** 合并所有分片的结果，使用 K-way Merge 取全局 Top-K

### 9.2 关键点

1. **分层合并**：
   - Segment → Shard → Collection
   - 每一层都进行 Top-K 选择

2. **结果数量**：
   - Segment 级别：返回 `top` 个（可能 oversampling）
   - Shard 级别：返回 `limit` 个（可能 undersampling）
   - Collection 级别：返回 `limit` 个

3. **合并算法**：
   - 使用 K-way Merge（多路归并）
   - 按分数/距离排序
   - 流式处理，内存高效

4. **性能优化**：
   - 并发搜索所有分片
   - Undersampling 优化（大 limit）
   - Oversampling 优化（量化场景）

### 9.3 代码位置

- **协调节点搜索**: `lib/collection/src/collection/search.rs:146-200`
- **结果合并**: `lib/collection/src/collection/search.rs:262-314`
- **分片搜索**: `lib/collection/src/shards/local_shard/search.rs`
- **Segment HNSW 搜索**: `lib/segment/src/index/hnsw_index/hnsw.rs:994-1145`
- **Undersampling 优化**: `lib/collection/src/collection/query.rs:91-144`

---

## 十、参考资料

- **HNSW 论文**: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (2016)
- **Qdrant 文档**: https://qdrant.tech/documentation/
- **代码位置**:
  - `lib/collection/src/collection/search.rs`
  - `lib/collection/src/collection/query.rs`
  - `lib/segment/src/index/hnsw_index/hnsw.rs`
