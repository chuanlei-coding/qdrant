# Qdrant 中 Undersampling 算法逻辑详解

## 概述

**Undersampling（欠采样）** 是 Qdrant 在分布式环境下对大量结果查询的一种性能优化策略。当用户请求的 `limit` 很大时（≥ 128），Qdrant 会利用数据随机分布的特性，让每个分片返回更少的结果，然后通过合并得到全局 Top-K，从而显著减少网络传输和内存占用。

---

## 一、问题背景

### 1.1 传统方法的局限性

在分布式环境下，如果用户请求 Top-1000，传统方法会让每个分片都返回 1000 个结果：

```
用户请求：Top-1000
分片数量：10
传统方法：
  - 每个分片返回：1000 个结果
  - 总传输量：10 × 1000 = 10,000 个结果
  - 合并后取 Top-1000
```

**问题**：
- 网络传输量大（10,000 个结果）
- 内存占用高（需要存储 10,000 个候选结果）
- 合并开销大（需要从 10,000 个候选中选择 Top-1000）

### 1.2 Undersampling 的优化思路

**核心思想**：利用数据随机分布的特性，每个分片只返回更少的结果，然后合并。

```
用户请求：Top-1000
分片数量：10
Undersampling 方法：
  - 每个分片返回：~171 个结果（而不是 1000 个）
  - 总传输量：10 × 171 = 1,710 个结果
  - 合并后取 Top-1000
```

**优势**：
- 网络传输量减少约 **83%**（从 10,000 降到 1,710）
- 内存占用减少约 **83%**
- 合并开销减少约 **83%**

**风险**：
- 可能错过一些真正的 Top-K 结果
- 但概率很低（因为数据随机分布）

---

## 二、算法原理

### 2.1 数学基础：Poisson 分布

Undersampling 算法基于 **Poisson 分布**（泊松分布）的概率估计。

#### 2.1.1 Poisson 分布简介

Poisson 分布用于描述在固定时间或空间内，随机事件发生的次数。其概率质量函数为：

```
P(X = k) = (λ^k × e^(-λ)) / k!
```

其中：
- `λ`（lambda）：平均发生次数
- `k`：实际发生次数

#### 2.1.2 在 Qdrant 中的应用

在 Qdrant 的分布式搜索场景中：
- **事件**：Top-K 结果中的某个点出现在某个分片中
- **λ（lambda）**：每个分片期望包含的 Top-K 结果数量
- **k**：每个分片实际需要返回的结果数量

**计算公式**：

```
λ = (用户请求的 limit) × (每个分片的概率) × (安全因子)
  = n × (1 / num_shards) × MORE_ENSURANCE_FACTOR
```

其中：
- `n`：用户请求的 `limit`
- `1 / num_shards`：每个分片的概率（假设数据均匀分布）
- `MORE_ENSURANCE_FACTOR = 1.2`：安全因子，增加可靠性

### 2.2 查找表生成

Qdrant 使用**预计算的查找表**来快速查找采样大小，而不是实时计算 Poisson 分布。

#### 2.2.1 查找表生成代码

**位置**: `lib/collection/src/collection_manager/probabilistic_search_sampling.rs:8-22`

```python
from scipy.stats import poisson
q = 0.999  # 概率：在所有分片中覆盖完整的 top 结果
res = []
for s in range(2, 1000):  # 分片数量
  for n in range(100, 10000, 50):  # top 参数
    lmbda = n * (1/s)  # 计算 lambda
    k = poisson.ppf(q**(1/s), lmbda)  # 计算采样大小
    res.append((lmbda, int(k)))
res = sorted(res, key=lambda x: x[0])  # 按 lambda 排序
# 去除重复项和 5% 以内的相似值
```

**关键参数**：
- `q = 0.999`：置信度，表示有 99.9% 的概率覆盖完整的 Top-K
- `q**(1/s)`：每个分片的置信度（因为需要所有分片都满足）

#### 2.2.2 查找表示例

**位置**: `lib/collection/src/collection_manager/probabilistic_search_sampling.rs:23-145`

```rust
const POISSON_DISTRIBUTION_SEARCH_SAMPLING: [(f64, usize); 121] = [
    (0.193, 4),      // lambda=0.193 → 采样大小=4
    (0.398, 5),      // lambda=0.398 → 采样大小=5
    (0.667, 6),      // lambda=0.667 → 采样大小=6
    // ...
    (125.0, 171),    // lambda=125.0 → 采样大小=171
    (150.0, 200),    // lambda=150.0 → 采样大小=200
    // ...
    (f64::MAX, usize::MAX),
];
```

**查找方法**：使用二分查找（`binary_search`）快速定位。

---

## 三、算法实现

### 3.1 核心函数：`modify_shard_query_for_undersampling_limits`

**位置**: `lib/collection/src/collection/query.rs:91-144`

#### 3.1.1 函数签名

```rust
fn modify_shard_query_for_undersampling_limits(
    batch_request: Arc<Vec<ShardQueryRequest>>,
    num_shards: usize,
    is_auto_sharding: bool,
) -> Arc<Vec<ShardQueryRequest>>
```

#### 3.1.2 关键常量

```rust
/// 如果查询 limit 超过此值，将进行 undersampling
const SHARD_QUERY_SUBSAMPLING_LIMIT: usize = 128;

/// 为 undersampling 提供额外的安全保障，
/// 检索更多点以防止 undersampling 错误。
/// 错误仍然可能发生，但概率足够低，可以接受。
const MORE_ENSURANCE_FACTOR: f64 = 1.2;
```

#### 3.1.3 完整代码解读

```rust
fn modify_shard_query_for_undersampling_limits(
    batch_request: Arc<Vec<ShardQueryRequest>>,
    num_shards: usize,
    is_auto_sharding: bool,
) -> Arc<Vec<ShardQueryRequest>> {
    // ========== 步骤 1：前置检查 ==========
    // 如果只有 1 个分片，不需要优化
    if num_shards <= 1 {
        return batch_request;
    }

    // 如果不是自动分片，不进行优化
    // 因为自动分片保证数据随机分布，而自定义分片可能不是
    if !is_auto_sharding {
        return batch_request;
    }

    // ========== 步骤 2：检查是否满足优化条件 ==========
    // 找到最大的 limit + offset
    let max_limit = batch_request
        .iter()
        .map(|req| req.limit + req.offset)
        .max()
        .unwrap_or(0);

    // 如果 limit < 128，不进行优化
    if max_limit < Self::SHARD_QUERY_SUBSAMPLING_LIMIT {
        return batch_request;
    }

    // ========== 步骤 3：为每个请求计算 undersample_limit ==========
    let mut new_requests = Vec::with_capacity(batch_request.len());

    for request in batch_request.iter() {
        let mut new_request = request.clone();
        let request_limit = new_request.limit + new_request.offset;

        // 如果是精确搜索，不进行优化
        let is_exact = request.params.as_ref().map(|p| p.exact).unwrap_or(false);

        if is_exact || request_limit < Self::SHARD_QUERY_SUBSAMPLING_LIMIT {
            new_requests.push(new_request);
            continue;
        }

        // ========== 步骤 4：计算 undersample_limit ==========
        // 示例：1000 limit, 10 shards
        // 1.0 / 10 * 1.2 = 0.12
        // lambda = 0.12 * 1000 = 120
        // 查找表：lambda=120 → 采样大小=171
        let undersample_limit = find_search_sampling_over_point_distribution(
            request_limit as f64,                    // n = 1000
            1. / num_shards as f64 * Self::MORE_ENSURANCE_FACTOR,  // p = 0.12
        );

        // 确保不超过原始 limit
        new_request.limit = std::cmp::min(undersample_limit, request_limit);
        new_request.offset = 0; // Offset 在集合级别处理
        new_requests.push(new_request);
    }

    Arc::new(new_requests)
}
```

### 3.2 查找函数：`find_search_sampling_over_point_distribution`

**位置**: `lib/collection/src/collection_manager/probabilistic_search_sampling.rs:148-156`

#### 3.2.1 函数实现

```rust
/// 使用二分查找找到给定 lambda 的采样大小
pub fn find_search_sampling_over_point_distribution(n: f64, p: f64) -> usize {
    // 计算目标 lambda
    let target_lambda = p * n;
    
    // 在查找表中二分查找
    let k = POISSON_DISTRIBUTION_SEARCH_SAMPLING
        .binary_search_by(|&(lambda, _sampling)| {
            lambda.partial_cmp(&target_lambda).unwrap()
        });
    
    match k {
        Ok(k) => POISSON_DISTRIBUTION_SEARCH_SAMPLING[k].1,  // 精确匹配
        Err(insert) => POISSON_DISTRIBUTION_SEARCH_SAMPLING[insert].1,  // 插入位置
    }
}
```

#### 3.2.2 计算示例

**场景**：用户请求 Top-1000，10 个分片

```
步骤 1：计算 p（每个分片的概率 × 安全因子）
  p = (1.0 / 10) * 1.2 = 0.12

步骤 2：计算 target_lambda
  target_lambda = 0.12 * 1000 = 120

步骤 3：在查找表中查找
  查找表中有：(125.0, 171)
  因为 120 < 125.0，所以选择 171

步骤 4：返回结果
  每个分片返回 171 个结果（而不是 1000 个）
```

---

## 四、完整流程示例

### 4.1 场景设置

```
用户请求：
  - limit: 1000
  - offset: 0
  - 查询类型：近似搜索（非精确）

系统配置：
  - 分片数量：10
  - 分片类型：自动分片（auto-sharding）
  - SHARD_QUERY_SUBSAMPLING_LIMIT: 128
  - MORE_ENSURANCE_FACTOR: 1.2
```

### 4.2 执行流程

```
1. 检查条件
   ├─ num_shards > 1? ✓ (10 > 1)
   ├─ is_auto_sharding? ✓ (是)
   └─ max_limit >= 128? ✓ (1000 >= 128)
   → 满足优化条件

2. 计算参数
   ├─ p = (1.0 / 10) * 1.2 = 0.12
   ├─ target_lambda = 0.12 * 1000 = 120
   └─ 查找表：lambda=120 → 采样大小=171

3. 修改请求
   ├─ 原始：每个分片 limit=1000
   └─ 优化后：每个分片 limit=171

4. 执行搜索
   ├─ 分片 1：返回 171 个结果
   ├─ 分片 2：返回 171 个结果
   ├─ ...
   └─ 分片 10：返回 171 个结果
   总候选结果：10 × 171 = 1,710

5. 合并结果
   ├─ K-way Merge：合并 1,710 个候选结果
   ├─ 去重：移除重复的点 ID
   └─ 取 Top-1000：返回全局 Top-1000

6. 返回给用户
   └─ 1,000 个结果
```

### 4.3 性能对比

| 指标 | 传统方法 | Undersampling | 改善 |
|------|---------|---------------|------|
| **每个分片返回** | 1,000 | 171 | **-83%** |
| **总传输量** | 10,000 | 1,710 | **-83%** |
| **内存占用** | 10,000 个点 | 1,710 个点 | **-83%** |
| **合并开销** | O(10,000 log 10) | O(1,710 log 10) | **-83%** |
| **准确性** | 100% | ~99.9% | 轻微下降 |

---

## 五、Undersampling 检测

### 5.1 检测机制

Qdrant 实现了 **Undersampling 检测**机制，用于验证优化是否成功。

#### 5.1.1 检测函数：`check_undersampling`

**位置**: `lib/collection/src/collection/query.rs:570-593`

```rust
/// 检查分片的最差结果是否出现在最终结果中。
/// 如果出现，说明可能存在 undersampling。
fn check_undersampling(
    &self,
    worst_merged_point: &ScoredPoint,      // 合并后最差的结果
    best_last_result: &ScoredPoint,        // 所有分片中最好的最后一个结果
    order: Order,
) {
    // 合并后的最差点应该比所有分片中最好的最后一个结果更好
    let is_properly_sampled = match order {
        Order::LargeBetter => {
            ScoredPointTies(worst_merged_point) > ScoredPointTies(best_last_result)
        }
        Order::SmallBetter => {
            ScoredPointTies(worst_merged_point) < ScoredPointTies(best_last_result)
        }
    };
    
    if !is_properly_sampled {
        log::debug!(
            "Undersampling detected. Collection: {}, Best last shard score: {}, Worst merged score: {}",
            self.id,
            best_last_result.score,
            worst_merged_point.score
        );
    }
}
```

#### 5.1.2 检测逻辑

**原理**：
- 如果 undersampling 成功，合并后的最差结果应该**优于**所有分片中最好的最后一个结果
- 如果合并后的最差结果**劣于**某个分片的最后一个结果，说明该分片可能还有更好的结果未被返回

**示例**：

```
分片 1 返回：171 个结果，最后一个分数 = 0.85
分片 2 返回：171 个结果，最后一个分数 = 0.82
分片 3 返回：171 个结果，最后一个分数 = 0.88  ← 最好的最后一个结果

合并后 Top-1000：
  - 最差结果分数 = 0.90  ✓ 正常（0.90 > 0.88）
  - 最差结果分数 = 0.80  ✗ 检测到 undersampling（0.80 < 0.88）
```

### 5.2 检测触发位置

**位置**: `lib/collection/src/collection/query.rs:664-671`

```rust
// 在合并结果后检测
if let Some(best_last_result) = best_last_result
    && number_of_shards > 1
    && is_enough  // 确保有足够的结果
{
    let worst_merged_point = merged.last();
    if let Some(worst_merged_point) = worst_merged_point {
        self.check_undersampling(worst_merged_point, &best_last_result, order);
    }
}
```

---

## 六、适用条件与限制

### 6.1 适用条件

Undersampling 优化**仅在以下条件全部满足时**才会启用：

1. **分片数量 > 1**
   ```rust
   if num_shards <= 1 {
       return batch_request;
   }
   ```

2. **自动分片（auto-sharding）**
   ```rust
   if !is_auto_sharding {
       return batch_request;
   }
   ```
   **原因**：自动分片保证数据随机分布，而自定义分片可能不是。

3. **limit ≥ 128**
   ```rust
   if max_limit < Self::SHARD_QUERY_SUBSAMPLING_LIMIT {  // 128
       return batch_request;
   }
   ```

4. **非精确搜索**
   ```rust
   let is_exact = request.params.as_ref().map(|p| p.exact).unwrap_or(false);
   if is_exact {
       // 不进行优化
   }
   ```

### 6.2 限制与风险

#### 6.2.1 准确性风险

- **概率**：有约 0.1% 的概率错过真正的 Top-K 结果
- **原因**：基于概率估计，不是 100% 保证
- **影响**：通常可以接受，因为向量索引本身也是近似的

#### 6.2.2 数据分布假设

- **假设**：数据在分片间随机分布
- **限制**：仅适用于自动分片，不适用于自定义分片
- **原因**：自定义分片可能按业务逻辑分布，不满足随机性

#### 6.2.3 重请求成本

- **问题**：如果 undersampling 失败，重新请求的成本很高
- **策略**：Qdrant "接受"这个风险，而不是重新请求
- **原因**：网络延迟和带宽成本远高于轻微准确性损失

---

## 七、算法复杂度分析

### 7.1 时间复杂度

| 操作 | 传统方法 | Undersampling | 说明 |
|------|---------|---------------|------|
| **查找表查询** | - | O(log 121) | 二分查找，常数时间 |
| **分片搜索** | O(K × log N) × S | O(k × log N) × S | k << K |
| **结果合并** | O(K × S × log S) | O(k × S × log S) | k << K |
| **总体** | O(K × S × log S) | O(k × S × log S) | **显著降低** |

其中：
- `K`：用户请求的 limit（如 1000）
- `k`：undersample_limit（如 171）
- `S`：分片数量（如 10）
- `N`：每个分片的向量数量

### 7.2 空间复杂度

| 指标 | 传统方法 | Undersampling | 改善 |
|------|---------|---------------|------|
| **网络传输** | K × S | k × S | **-83%** |
| **内存占用** | K × S | k × S | **-83%** |

### 7.3 实际性能提升

**示例**：Top-1000，10 个分片

```
传统方法：
  - 传输：10,000 个结果
  - 内存：10,000 个点
  - 合并时间：~10ms

Undersampling：
  - 传输：1,710 个结果（-83%）
  - 内存：1,710 个点（-83%）
  - 合并时间：~2ms（-80%）
```

---

## 八、代码位置总结

### 8.1 核心文件

| 文件 | 功能 |
|------|------|
| `lib/collection/src/collection/query.rs` | Undersampling 主逻辑 |
| `lib/collection/src/collection_manager/probabilistic_search_sampling.rs` | Poisson 分布查找表 |

### 8.2 关键函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `modify_shard_query_for_undersampling_limits` | `query.rs:91` | 修改分片查询的 limit |
| `find_search_sampling_over_point_distribution` | `probabilistic_search_sampling.rs:148` | 查找采样大小 |
| `check_undersampling` | `query.rs:570` | 检测 undersampling 是否成功 |

### 8.3 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `SHARD_QUERY_SUBSAMPLING_LIMIT` | 128 | 启用优化的最小 limit |
| `MORE_ENSURANCE_FACTOR` | 1.2 | 安全因子 |
| `POISSON_DISTRIBUTION_SEARCH_SAMPLING` | 121 项 | 预计算的查找表 |

---

## 九、总结

### 9.1 核心思想

Undersampling 利用**数据随机分布**和**Poisson 分布概率估计**，在保证高准确性的同时，显著减少网络传输和内存占用。

### 9.2 关键优势

1. **性能提升**：网络传输和内存占用减少约 83%
2. **准确性保证**：99.9% 的置信度覆盖完整的 Top-K
3. **自动优化**：无需用户配置，自动启用

### 9.3 适用场景

- ✅ 大 limit 查询（≥ 128）
- ✅ 自动分片模式
- ✅ 近似搜索（非精确搜索）
- ❌ 精确搜索
- ❌ 自定义分片

### 9.4 设计权衡

Qdrant 选择了**性能优先**的策略：
- 接受 0.1% 的准确性损失
- 换取 83% 的性能提升
- 不进行重请求（因为成本太高）

这种权衡在大多数场景下是合理的，因为：
1. 向量搜索本身就是近似的
2. 0.1% 的准确性损失通常可以接受
3. 性能提升非常显著

---

## 参考资料

- [分布式环境下 HNSW 向量搜索工作原理](./分布式环境下HNSW向量搜索工作原理.md)
- [Poisson Distribution - Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Qdrant 协调节点实现解读](./Qdrant协调节点实现解读.md)
