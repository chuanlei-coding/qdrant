# Qdrant 协调节点实现解读

## 概述

在 Qdrant 的分布式架构中，**协调节点（Coordinator）**并不是一个独立的组件，而是通过多个层次的协作来实现的。本文档将详细解读协调节点的实现机制和代码位置。

---

## 一、协调节点的层次结构

Qdrant 的协调功能分布在以下几个层次：

```
客户端请求
    ↓
Dispatcher（请求路由层）
    ↓
TableOfContent（集合管理器）
    ↓
Collection（集合层 - 主要协调节点）
    ↓
ShardHolder（分片持有者）
    ↓
ShardReplicaSet（副本集）
    ↓
LocalShard / RemoteShard（实际分片）
```

### 1.1 各层职责

| 层次 | 组件 | 职责 |
|------|------|------|
| **请求路由层** | `Dispatcher` | 接收外部请求，路由到相应的 Collection |
| **集合管理层** | `TableOfContent` | 管理所有 Collection 实例 |
| **协调节点层** | `Collection` | **核心协调逻辑**：选择分片、分发请求、合并结果 |
| **分片管理层** | `ShardHolder` | 管理分片映射，选择目标分片 |
| **副本管理层** | `ShardReplicaSet` | 管理本地和远程分片副本 |
| **执行层** | `LocalShard` / `RemoteShard` | 实际执行搜索操作 |

---

## 二、Dispatcher：请求路由层

### 2.1 代码位置

**文件**: `lib/storage/src/dispatcher.rs`

### 2.2 核心结构

```rust
#[derive(Clone)]
pub struct Dispatcher {
    toc: Arc<TableOfContent>,              // 集合管理器
    consensus_state: Option<ConsensusStateRef>,  // 共识状态（集群模式）
    resharding_enabled: bool,               // 是否启用重分片
}
```

### 2.3 主要功能

1. **请求路由**：将外部请求路由到相应的 Collection
2. **集合元操作**：处理集合的创建、删除、更新等操作
3. **集群状态管理**：在分布式模式下管理集群状态

### 2.4 初始化位置

**文件**: `src/main.rs:402`

```rust
// Router for external queries.
// It decides if query should go directly to the ToC or through the consensus.
let mut dispatcher = Dispatcher::new(toc_arc.clone());
```

---

## 三、TableOfContent：集合管理器

### 3.1 代码位置

**文件**: `lib/storage/src/content_manager/toc.rs`（推测，未直接找到）

### 3.2 主要功能

1. **集合管理**：管理所有 Collection 实例的生命周期
2. **集合查找**：根据集合名称查找对应的 Collection
3. **存储配置**：管理存储路径和配置

### 3.3 初始化位置

**文件**: `src/main.rs:374-383`

```rust
// Table of content manages the list of collections.
// It is a main entry point for the storage.
let toc = TableOfContent::new(
    &settings.storage,
    search_runtime,
    update_runtime,
    general_runtime,
    optimizer_resource_budget,
    channel_service.clone(),
    persistent_consensus_state.this_peer_id(),
    propose_operation_sender.clone(),
);
```

---

## 四、Collection：核心协调节点

### 4.1 代码位置

**文件**: `lib/collection/src/collection/mod.rs`

### 4.2 核心结构

```rust
pub struct Collection {
    pub(crate) id: CollectionId,
    pub(crate) shards_holder: Arc<LockedShardHolder>,  // 分片持有者
    pub(crate) collection_config: Arc<RwLock<CollectionConfigInternal>>,
    pub(crate) shared_storage_config: Arc<SharedStorageConfig>,
    this_peer_id: PeerId,
    channel_service: ChannelService,
    // ... 其他字段
}
```

### 4.3 协调逻辑：`do_core_search_batch`

**文件**: `lib/collection/src/collection/search.rs:146-200`

这是协调节点的**核心方法**，负责：

1. **选择目标分片**
2. **并发查询所有分片**
3. **合并结果**

#### 4.3.1 完整代码解读

```rust
async fn do_core_search_batch(
    &self,
    request: CoreSearchRequestBatch,
    read_consistency: Option<ReadConsistency>,
    shard_selection: &ShardSelectorInternal,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
    let request = Arc::new(request);
    let instant = Instant::now();

    // ========== 步骤 1：选择目标分片 ==========
    // query all shards concurrently
    let all_searches_res = {
        let shard_holder = self.shards_holder.read().await;
        
        // 根据 shard_selection 选择目标分片
        // 可能的选择：
        // - ShardSelectorInternal::All：所有分片
        // - ShardSelectorInternal::ShardId(id)：指定分片
        // - ShardSelectorInternal::ShardKey(key)：指定分片键
        let target_shards = shard_holder.select_shards(shard_selection)?;
        
        // ========== 步骤 2：并发查询所有分片 ==========
        let all_searches = target_shards.into_iter().map(|(shard, shard_key)| {
            let shard_key = shard_key.cloned();
            
            // 对每个分片调用 core_search
            // 注意：这里 shard 是 ShardReplicaSet，它会处理本地/远程分片的选择
            shard
                .core_search(
                    request.clone(),
                    read_consistency,
                    shard_selection.is_shard_id(),
                    timeout,
                    hw_measurement_acc.clone(),
                )
                .and_then(move |mut records| async move {
                    // 为结果添加 shard_key 信息
                    if shard_key.is_none() {
                        return Ok(records);
                    }
                    for batch in &mut records {
                        for point in batch {
                            point.shard_key.clone_from(&shard_key);
                        }
                    }
                    Ok(records)
                })
        });
        
        // 使用 try_join_all 并发执行所有分片的搜索
        // 总搜索时间 ≈ 最慢分片的搜索时间
        future::try_join_all(all_searches).await?
    };

    // ========== 步骤 3：合并所有分片的结果 ==========
    let result = self
        .merge_from_shards(
            all_searches_res,
            request.clone(),
            !shard_selection.is_shard_id(),
        )
        .await;

    // 记录慢查询
    let filters_refs = request.searches.iter().map(|req| req.filter.as_ref());
    self.post_process_if_slow_request(instant.elapsed(), filters_refs);

    result
}
```

#### 4.3.2 关键点解析

1. **并发查询**：使用 `future::try_join_all` 并发查询所有分片，而不是串行查询
2. **分片选择**：通过 `shard_holder.select_shards()` 选择目标分片
3. **结果合并**：通过 `merge_from_shards()` 合并所有分片的结果

---

## 五、ShardHolder：分片选择逻辑

### 5.1 代码位置

**文件**: `lib/collection/src/shards/shard_holder/mod.rs`

### 5.2 核心方法：`select_shards`

**位置**: `lib/collection/src/shards/shard_holder/mod.rs:576-661`

#### 5.2.1 代码解读

```rust
pub fn select_shards<'a>(
    &'a self,
    shard_selector: &'a ShardSelectorInternal,
) -> CollectionResult<Vec<(&'a ShardReplicaSet, Option<&'a ShardKey>)>> {
    let mut res = Vec::new();

    match shard_selector {
        // ========== 情况 1：查询所有分片 ==========
        ShardSelectorInternal::All => {
            let is_custom_sharding = match self.sharding_method {
                ShardingMethod::Auto => false,
                ShardingMethod::Custom => true,
            };

            for (&shard_id, shard) in self.shards.iter() {
                // 忽略正在迁移的分片
                let resharding_migrating_up = /* ... */;
                if resharding_migrating_up {
                    continue;
                }

                // 对于自定义分片，跳过非活跃分片
                if is_custom_sharding && !shard.shard_is_active() {
                    continue;
                }

                let shard_key = self.shard_id_to_key_mapping.get(&shard_id);
                res.push((shard, shard_key));
            }
        }
        
        // ========== 情况 2：根据分片键选择 ==========
        ShardSelectorInternal::ShardKey(shard_key) => {
            // 根据 shard_key 查找对应的 shard_id
            for shard_id in self.get_shard_ids_by_key(shard_key)? {
                if let Some(replica_set) = self.shards.get(&shard_id) {
                    res.push((replica_set, Some(shard_key)));
                }
            }
        }
        
        // ========== 情况 3：根据分片 ID 选择 ==========
        ShardSelectorInternal::ShardId(shard_id) => {
            if let Some(replica_set) = self.shards.get(shard_id) {
                res.push((replica_set, self.shard_id_to_key_mapping.get(shard_id)));
            } else {
                return Err(shard_not_found_error(*shard_id));
            }
        }
        
        // ... 其他情况
    }
    
    Ok(res)
}
```

#### 5.2.2 选择策略

| 选择器类型 | 说明 | 使用场景 |
|-----------|------|---------|
| `All` | 选择所有活跃分片 | 全局搜索 |
| `ShardKey(key)` | 根据分片键选择 | 多租户场景 |
| `ShardId(id)` | 根据分片 ID 选择 | 指定分片搜索 |
| `ShardKeys(keys)` | 根据多个分片键选择 | 多租户批量查询 |

---

## 六、ShardReplicaSet：副本管理

### 6.1 代码位置

**文件**: `lib/collection/src/shards/replica_set/mod.rs`

### 6.2 核心结构

```rust
pub struct ShardReplicaSet {
    local: RwLock<Option<Shard>>,      // 本地分片（如果存在）
    remotes: RwLock<Vec<RemoteShard>>, // 远程分片列表
    replica_state: Arc<SaveOnDisk<ReplicaSetState>>,
    shard_id: ShardId,
    shard_key: Option<ShardKey>,
    // ... 其他字段
}
```

### 6.3 读取操作执行：`execute_and_resolve_read_operation`

**文件**: `lib/collection/src/shards/replica_set/execute_read_operation.rs:42-126`

#### 6.3.1 代码解读

```rust
pub async fn execute_and_resolve_read_operation<Res, F>(
    &self,
    read_operation: F,
    read_consistency: Option<ReadConsistency>,
    local_only: bool,
) -> CollectionResult<Res>
where
    F: Fn(&(dyn ShardOperation + Send + Sync)) -> BoxFuture<'_, CollectionResult<Res>>,
    Res: Resolve,
{
    // ========== 情况 1：仅本地查询 ==========
    if local_only {
        return self.execute_local_read_operation(read_operation).await;
    }

    let read_consistency = read_consistency.unwrap_or_default();

    // ========== 统计副本状态 ==========
    let local_count = usize::from(self.peer_state(self.this_peer_id()).is_some());
    let active_local_count = usize::from(self.peer_is_readable(self.this_peer_id()));
    let remotes = self.remotes.read().await;
    let remotes_count = remotes.len();
    let active_remotes_count = remotes
        .iter()
        .filter(|remote| self.peer_is_readable(remote.peer_id))
        .count();

    let total_count = local_count + remotes_count;
    let active_count = active_local_count + active_remotes_count;

    // ========== 根据一致性要求计算需要的成功结果数 ==========
    let (mut required_successful_results, condition) = match read_consistency {
        ReadConsistency::Type(ReadConsistencyType::All) => {
            (total_count, ResolveCondition::All)
        }
        ReadConsistency::Type(ReadConsistencyType::Majority) => {
            (total_count, ResolveCondition::Majority)
        }
        ReadConsistency::Type(ReadConsistencyType::Quorum) => {
            (total_count / 2 + 1, ResolveCondition::All)
        }
        ReadConsistency::Factor(factor) => {
            (factor.clamp(1, total_count), ResolveCondition::All)
        }
    };

    // ========== 执行集群读取操作 ==========
    let mut responses = self
        .execute_cluster_read_operation(
            read_operation,
            required_successful_results,
            Some(remotes),
        )
        .await?;

    // ========== 解析结果 ==========
    if responses.is_empty() {
        Ok(Res::default())
    } else if responses.len() == 1 {
        Ok(responses.pop().unwrap())
    } else {
        // 多个结果需要解析（根据一致性要求）
        Ok(Res::resolve(responses, condition))
    }
}
```

### 6.4 集群读取操作：`execute_cluster_read_operation`

**文件**: `lib/collection/src/shards/replica_set/execute_read_operation.rs:147-328`

#### 6.4.1 关键逻辑

```rust
async fn execute_cluster_read_operation<Res, F>(
    &self,
    read_operation: F,
    required_successful_results: usize,
    remotes: Option<tokio::sync::RwLockReadGuard<'_, Vec<RemoteShard>>>,
) -> CollectionResult<Vec<Res>>
{
    // ========== 1. 尝试获取本地分片 ==========
    let local_read = self.local.try_read().ok();
    let local_read = local_read.as_ref().and_then(|local| local.as_ref());

    let (local, is_local_ready, update_watcher) = match local_read {
        Some(local) => {
            let update_watcher = local.watch_for_update();
            let is_local_ready = !local.is_update_in_progress();
            (Some(local), is_local_ready, Some(update_watcher))
        }
        None => (None, false, None),
    };

    // ========== 2. 准备本地操作 ==========
    let local_is_readable = self.peer_is_readable(self.this_peer_id());
    let local_operation = if local_is_readable {
        let local_operation = async {
            let Some(local) = local else {
                return Err(CollectionError::service_error(format!(
                    "Local shard {} not found",
                    self.shard_id,
                )));
            };
            read_operation(local.get()).await
        };
        Some(local_operation.map(|result| (result, true)).left_future())
    } else {
        None
    };

    // ========== 3. 准备远程操作 ==========
    let mut readable_remotes: Vec<_> = remotes
        .iter()
        .filter(|remote| self.peer_is_readable(remote.peer_id))
        .collect();

    readable_remotes.shuffle(&mut rand::rng());  // 随机打乱，负载均衡

    let remote_operations = readable_remotes.into_iter().map(|remote| {
        read_operation(remote)
            .map(|result| (result, false))
            .right_future()
    });

    // ========== 4. 合并本地和远程操作 ==========
    let mut operations = local_operation.into_iter().chain(remote_operations);

    // ========== 5. 确定扇出策略 ==========
    // 如果本地可用且就绪，默认扇出为 0（仅本地）
    // 否则，默认扇出为 1（至少查询一个远程）
    let default_fan_out = if is_local_ready && local_is_readable {
        0
    } else {
        1
    };

    let read_fan_out_factor: usize = self
        .collection_config
        .read()
        .await
        .params
        .read_fan_out_factor
        .unwrap_or(default_fan_out);

    // ========== 6. 执行操作并收集结果 ==========
    // 优先使用本地，然后根据 fan_out_factor 选择远程副本
    // ... 具体实现
}
```

#### 6.4.2 关键策略

1. **本地优先**：如果本地分片可用且就绪，优先使用本地分片
2. **扇出策略**：根据 `read_fan_out_factor` 决定查询多少个远程副本
3. **负载均衡**：随机打乱远程分片顺序，实现负载均衡
4. **一致性保证**：根据 `read_consistency` 要求，确保足够的成功结果

---

## 七、结果合并：`merge_from_shards`

### 7.1 代码位置

**文件**: `lib/collection/src/collection/search.rs:262-314`

### 7.2 代码解读

```rust
async fn merge_from_shards(
    &self,
    mut all_searches_res: Vec<Vec<Vec<ScoredPoint>>>,  // 所有分片的结果
    request: Arc<CoreSearchRequestBatch>,
    is_client_request: bool,
) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
    let batch_size = request.searches.len();
    let collection_params = self.collection_config.read().await.params.clone();

    // ========== 为每个搜索请求合并结果 ==========
    let mut top_results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(batch_size);
    let mut seen_ids = AHashSet::new();  // 用于去重

    for (batch_index, request) in request.searches.iter().enumerate() {
        // ========== 1. 确定排序顺序 ==========
        let order = if request.query.is_distance_scored() {
            collection_params
                .get_distance(request.query.get_vector_name())?
                .distance_order()  // 根据距离类型确定排序（LargeBetter 或 SmallBetter）
        } else {
            Order::LargeBetter
        };

        // ========== 2. 提取当前批次的所有分片结果 ==========
        let results_from_shards = all_searches_res
            .iter_mut()
            .map(|res| res.get_mut(batch_index).map_or(Vec::new(), mem::take));

        // ========== 3. K-way Merge ==========
        // 使用 itertools 的 kmerge_by 进行 K 路归并
        let merged_iter = match order {
            Order::LargeBetter => Either::Left(
                results_from_shards.kmerge_by(|a, b| a > b)  // 降序
            ),
            Order::SmallBetter => Either::Right(
                results_from_shards.kmerge_by(|a, b| a < b)  // 升序
            ),
        }
        .filter(|point| seen_ids.insert(point.id));  // 去重

        // ========== 4. 应用 offset 和 limit ==========
        let top_res = if is_client_request && request.offset > 0 {
            // 客户端请求：跳过 offset，取 limit
            merged_iter
                .skip(request.offset)
                .take(request.limit)
                .collect()
        } else {
            // 内部请求：取 offset + limit
            merged_iter.take(request.offset + request.limit).collect()
        };

        top_results.push(top_res);
        seen_ids.clear();  // 为下一个批次清空
    }

    Ok(top_results)
}
```

### 7.3 关键点解析

1. **K-way Merge**：使用 `itertools::kmerge_by` 进行 K 路归并，时间复杂度 O(N log K)，其中 N 是总结果数，K 是分片数
2. **去重**：使用 `AHashSet` 基于点 ID 去重
3. **排序**：根据距离类型（Cosine/Dot Product vs Euclidean）确定排序顺序
4. **分页**：正确处理 `offset` 和 `limit`

---

## 八、完整搜索流程示例

### 8.1 场景：全局搜索 Top-10

```
用户请求: search(limit=10)
    ↓
Dispatcher::toc() → TableOfContent::get_collection()
    ↓
Collection::core_search_batch()
    ↓
Collection::do_core_search_batch()
    ↓
[步骤 1] ShardHolder::select_shards(ShardSelectorInternal::All)
    → 返回: [ShardReplicaSet(0), ShardReplicaSet(1), ShardReplicaSet(2)]
    ↓
[步骤 2] 并发查询所有分片
    ├─ ShardReplicaSet(0)::core_search()
    │   └─ execute_and_resolve_read_operation()
    │       └─ execute_cluster_read_operation()
    │           ├─ LocalShard::do_search() → 返回 10 个结果
    │           └─ (可选) RemoteShard::search() → 返回 10 个结果
    │
    ├─ ShardReplicaSet(1)::core_search()
    │   └─ ... → 返回 10 个结果
    │
    └─ ShardReplicaSet(2)::core_search()
        └─ ... → 返回 10 个结果
    ↓
[步骤 3] Collection::merge_from_shards()
    ├─ 输入: [
    │     [10 个结果 from Shard 0],
    │     [10 个结果 from Shard 1],
    │     [10 个结果 from Shard 2]
    │   ]
    ├─ K-way Merge: 合并 30 个结果，按分数排序
    ├─ 去重: 移除重复的点 ID
    └─ 取 Top-10: 返回全局 Top-10
    ↓
返回给用户: 10 个结果
```

### 8.2 代码调用链

```
src/actix/api/search_api.rs::search_points()
    ↓
crate::common::query::do_core_search_points()
    ↓
Dispatcher::toc().core_search_batch()
    ↓
TableOfContent::core_search_batch()
    ↓
Collection::core_search_batch()  [lib/collection/src/collection/search.rs:51]
    ↓
Collection::do_core_search_batch()  [lib/collection/src/collection/search.rs:146]
    ├─ ShardHolder::select_shards()  [lib/collection/src/shards/shard_holder/mod.rs:576]
    ├─ ShardReplicaSet::core_search()  [并发]
    │   └─ ShardReplicaSet::execute_and_resolve_read_operation()
    │       └─ ShardReplicaSet::execute_cluster_read_operation()
    │           ├─ LocalShard::do_search()  [lib/collection/src/shards/local_shard/search.rs:30]
    │           └─ RemoteShard::search()  [lib/collection/src/shards/remote_shard.rs]
    └─ Collection::merge_from_shards()  [lib/collection/src/collection/search.rs:262]
```

---

## 九、关键设计要点

### 9.1 并发查询

**优势**：
- 总搜索时间 ≈ 最慢分片的搜索时间
- 而不是所有分片搜索时间的总和

**实现**：
```rust
future::try_join_all(all_searches).await?
```

### 9.2 本地优先策略

**优势**：
- 减少网络延迟
- 降低网络带宽消耗
- 提高查询性能

**实现**：
```rust
let default_fan_out = if is_local_ready && local_is_readable {
    0  // 仅本地
} else {
    1  // 至少一个远程
};
```

### 9.3 K-way Merge 优化

**优势**：
- 流式处理，内存效率高
- 只需要维护一个大小为 `limit` 的堆
- 时间复杂度 O(N log K)

**实现**：
```rust
results_from_shards.kmerge_by(|a, b| a > b)
    .filter(|point| seen_ids.insert(point.id))
    .take(request.offset + request.limit)
```

### 9.4 去重机制

**原因**：
- 同一个点可能存在于多个分片（在迁移过程中）
- 需要确保结果中每个点只出现一次

**实现**：
```rust
let mut seen_ids = AHashSet::new();
.filter(|point| seen_ids.insert(point.id))
```

---

## 十、总结

### 10.1 协调节点的本质

Qdrant 的协调节点不是单一组件，而是通过以下层次协作实现：

1. **Dispatcher**：请求路由
2. **TableOfContent**：集合管理
3. **Collection**：**核心协调逻辑**
4. **ShardHolder**：分片选择
5. **ShardReplicaSet**：副本管理

### 10.2 核心协调逻辑位置

**主要文件**：
- `lib/collection/src/collection/search.rs` - Collection 的搜索协调逻辑
- `lib/collection/src/shards/shard_holder/mod.rs` - 分片选择逻辑
- `lib/collection/src/shards/replica_set/execute_read_operation.rs` - 副本读取逻辑

### 10.3 关键方法

| 方法 | 位置 | 功能 |
|------|------|------|
| `do_core_search_batch` | `collection/search.rs:146` | 协调节点核心方法 |
| `select_shards` | `shard_holder/mod.rs:576` | 分片选择 |
| `execute_and_resolve_read_operation` | `replica_set/execute_read_operation.rs:42` | 副本读取执行 |
| `merge_from_shards` | `collection/search.rs:262` | 结果合并 |

### 10.4 性能优化

1. **并发查询**：所有分片并发搜索
2. **本地优先**：优先使用本地分片
3. **K-way Merge**：高效的结果合并
4. **去重优化**：基于哈希集合的快速去重

---

## 参考资料

- [分布式环境下 HNSW 向量搜索工作原理](./分布式环境下HNSW向量搜索工作原理.md)
- [系统架构分析文档](./系统架构分析文档.md)
