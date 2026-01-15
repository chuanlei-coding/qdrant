# Collection、Shard、Segment 关系详解

## 概述

在 Qdrant 的架构中，**Collection（集合）**、**Shard（分片）** 和 **Segment（段）** 是三个层次的数据组织单位，它们之间是**包含关系**：Collection 包含多个 Shard，Shard 包含多个 Segment。

---

## 一、层次关系

### 1.1 基本关系

```
Collection（集合）
    ├─ Shard 0（分片 0）
    │   ├─ Segment 0（段 0）
    │   ├─ Segment 1（段 1）
    │   └─ Segment 2（段 2）
    ├─ Shard 1（分片 1）
    │   ├─ Segment 3（段 3）
    │   └─ Segment 4（段 4）
    └─ Shard 2（分片 2）
        ├─ Segment 5（段 5）
        └─ Segment 6（段 6）
```

### 1.2 关系总结

| 层次 | 组件 | 包含关系 | 作用范围 |
|------|------|---------|---------|
| **第 1 层** | Collection | 包含多个 Shard | 逻辑集合，用户可见 |
| **第 2 层** | Shard | 包含多个 Segment | 分布式分片，可跨节点 |
| **第 3 层** | Segment | 实际存储数据 | 物理存储单位，不可跨节点 |

---

## 二、Collection（集合）

### 2.1 定义

**Collection** 是用户可见的**逻辑集合**，是 Qdrant 中数据组织的最高层次。

### 2.2 代码位置

**文件**: `lib/collection/src/collection/mod.rs`

### 2.3 核心结构

```62:94:lib/collection/src/collection/mod.rs
/// Collection's data is split into several shards.
pub struct Collection {
    pub(crate) id: CollectionId,
    pub(crate) shards_holder: Arc<LockedShardHolder>,  // 分片持有者
    pub(crate) collection_config: Arc<RwLock<CollectionConfigInternal>>,
    pub(crate) shared_storage_config: Arc<SharedStorageConfig>,
    payload_index_schema: Arc<SaveOnDisk<PayloadIndexSchema>>,
    optimizers_overwrite: Option<OptimizersConfigDiff>,
    this_peer_id: PeerId,
    path: PathBuf,
    snapshots_path: PathBuf,
    channel_service: ChannelService,
    transfer_tasks: Mutex<TransferTasksPool>,
    request_shard_transfer_cb: RequestShardTransfer,
    notify_peer_failure_cb: ChangePeerFromState,
    abort_shard_transfer_cb: replica_set::AbortShardTransfer,
    init_time: Duration,
    // One-way boolean flag that is set to true when the collection is fully initialized
    // i.e. all shards are activated for the first time.
    is_initialized: Arc<IsReady>,
    // Lock to temporary block collection update operations while the collection is being migrated.
    // Lock is acquired for read on update operation and can be acquired for write externally,
    // which will block all update operations until the lock is released.
    updates_lock: Arc<RwLock<()>>,
    // Update runtime handle.
    update_runtime: Handle,
    // Search runtime handle.
    search_runtime: Handle,
    optimizer_resource_budget: ResourceBudget,
    // Cached statistics of collection size, may be outdated.
    collection_stats_cache: CollectionSizeStatsCache,
    // Background tasks to clean shards
    shard_clean_tasks: ShardCleanTasks,
}
```

### 2.4 关键字段

- **`shards_holder`**: `Arc<LockedShardHolder>` - 管理所有分片的容器
- **`id`**: `CollectionId` - 集合的唯一标识符
- **`collection_config`**: 集合的配置信息

### 2.5 主要职责

1. **分片管理**：通过 `ShardHolder` 管理所有分片
2. **请求协调**：接收用户请求，分发到相应的分片
3. **结果合并**：合并来自多个分片的搜索结果
4. **配置管理**：管理集合级别的配置

---

## 三、Shard（分片）

### 3.1 定义

**Shard** 是 Collection 的**水平分片**，用于实现分布式存储和查询。一个 Collection 的数据被分割到多个 Shard 中，每个 Shard 可以位于不同的节点上。

### 3.2 代码位置

**文件**: `lib/collection/src/shards/shard.rs`

### 3.3 核心结构

```45:54:lib/collection/src/shards/shard.rs
/// Shard
///
/// Contains a part of the collection's points
pub enum Shard {
    Local(LocalShard),
    Proxy(ProxyShard),
    ForwardProxy(ForwardProxyShard),
    QueueProxy(QueueProxyShard),
    Dummy(DummyShard),
}
```

### 3.4 Shard 类型

| 类型 | 说明 | 位置 |
|------|------|------|
| **LocalShard** | 本地分片，包含实际数据 | 当前节点 |
| **ProxyShard** | 代理分片，指向远程分片 | 当前节点（代理） |
| **ForwardProxyShard** | 转发代理分片 | 当前节点（转发） |
| **QueueProxyShard** | 队列代理分片 | 当前节点（队列） |
| **DummyShard** | 虚拟分片 | 占位符 |

### 3.5 LocalShard 结构

**文件**: `lib/collection/src/shards/local_shard/mod.rs`

```94:136:lib/collection/src/shards/local_shard/mod.rs
/// LocalShard
///
/// LocalShard is an entity that can be moved between peers and contains some part of one collections data.
///
/// Holds all object, required for collection functioning
#[must_use = "Local Shard must be explicitly handled"]
pub struct LocalShard {
    collection_name: CollectionId,
    pub(super) segments: LockedSegmentHolder,  // 段持有者
    pub(super) collection_config: Arc<TokioRwLock<CollectionConfigInternal>>,
    pub(super) shared_storage_config: Arc<SharedStorageConfig>,
    pub(crate) payload_index_schema: Arc<SaveOnDisk<PayloadIndexSchema>>,
    pub(super) wal: RecoverableWal,
    pub(super) update_handler: Arc<Mutex<UpdateHandler>>,
    pub(super) update_sender: ArcSwap<Sender<UpdateSignal>>,
    pub(super) update_tracker: UpdateTracker,
    pub(super) path: PathBuf,
    pub(super) optimizers: Arc<Vec<Arc<Optimizer>>>,
    pub(super) optimizers_log: Arc<ParkingMutex<TrackerLog>>,
    pub(super) total_optimized_points: Arc<AtomicUsize>,
    pub(super) search_runtime: Handle,
    disk_usage_watcher: DiskUsageWatcher,
    read_rate_limiter: Option<ParkingMutex<RateLimiter>>,

    is_gracefully_stopped: bool,

    /// Update operation lock
    /// The lock, which must prevent updates critical sections of other operations, which
    /// are not compatible with updates.
    ///
    /// Currently used for:
    ///
    /// * Blocking updates during scroll + retrieve operations
    ///   Consistency of scroll operations is especially important for internal processes like
    ///   re-sharding and shard transfer, so explicit lock for those operations is required.
    ///
    /// * Blocking updates during some parts of snapshot creation
    ///   Snapshotting process wraps and unwraps proxy segments, which might
    ///   create inconsistencies if updates are applied concurrently.
    ///
    /// Write lock must be held for updates, while read lock must be held for critical sections
    pub(super) update_operation_lock: Arc<tokio::sync::RwLock<()>>,
}
```

### 3.6 关键字段

- **`segments`**: `LockedSegmentHolder` - 管理所有段的容器
- **`wal`**: `RecoverableWal` - 预写日志，用于持久化
- **`path`**: `PathBuf` - 分片的存储路径

### 3.7 主要职责

1. **段管理**：通过 `SegmentHolder` 管理所有段
2. **数据持久化**：通过 WAL 确保数据持久化
3. **优化器管理**：管理段的优化操作（合并、删除等）
4. **搜索执行**：在分片级别执行搜索，合并段的结果

---

## 四、Segment（段）

### 4.1 定义

**Segment** 是**实际存储数据的基本单位**，是 Qdrant 中数据存储的最底层。每个 Segment 包含：
- 向量数据（Vector Storage）
- 向量索引（Vector Index，如 HNSW）
- 载荷数据（Payload Storage）
- 载荷索引（Payload Index）
- ID 映射（ID Tracker）

### 4.2 代码位置

**文件**: `lib/segment/src/segment/mod.rs`

### 4.3 核心结构

```57:93:lib/segment/src/segment/mod.rs
/// Segment - an object which manages an independent group of points.
///
/// - Provides storage, indexing and managing operations for points (vectors + payload)
/// - Keeps track of point versions
/// - Persists data
/// - Keeps track of occurred errors
#[derive(Debug)]
pub struct Segment {
    /// Initial version this segment was created at
    pub initial_version: Option<SeqNumberType>,
    /// Latest update operation number, applied to this segment
    /// If None, there were no updates and segment is empty
    pub version: Option<SeqNumberType>,
    /// Latest persisted version
    /// Locked structure on which we hold the lock during flush to prevent concurrent flushes
    pub persisted_version: Arc<Mutex<Option<SeqNumberType>>>,
    /// Lock to prevent concurrent flushes and used for waiting for ongoing flushes to finish.
    pub is_alive_flush_lock: IsAliveLock,
    /// Path of the storage root
    pub current_path: PathBuf,
    pub version_tracker: VersionTracker,
    /// Component for mapping external ids to internal and also keeping track of point versions
    pub id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
    pub vector_data: HashMap<VectorNameBuf, VectorData>,
    pub payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    pub payload_storage: Arc<AtomicRefCell<PayloadStorageEnum>>,
    /// Shows if it is possible to insert more points into this segment
    pub appendable_flag: bool,
    /// Shows what kind of indexes and storages are used in this segment
    pub segment_type: SegmentType,
    pub segment_config: SegmentConfig,
    /// Last unhandled error
    /// If not None, all update operations will be aborted until original operation is performed properly
    pub error_status: Option<SegmentFailedState>,
    #[cfg(feature = "rocksdb")]
    pub database: Option<Arc<parking_lot::RwLock<DB>>>,
}
```

### 4.4 关键字段

- **`vector_data`**: `HashMap<VectorNameBuf, VectorData>` - 向量数据存储
- **`payload_storage`**: `PayloadStorageEnum` - 载荷存储
- **`payload_index`**: `StructPayloadIndex` - 载荷索引
- **`id_tracker`**: `IdTrackerSS` - ID 映射和版本跟踪
- **`appendable_flag`**: `bool` - 是否可追加数据

### 4.5 VectorData 结构

```95:99:lib/segment/src/segment/mod.rs
pub struct VectorData {
    pub vector_index: Arc<AtomicRefCell<VectorIndexEnum>>,
    pub vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    pub quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
}
```

### 4.6 主要职责

1. **数据存储**：存储向量和载荷数据
2. **索引管理**：维护向量索引（HNSW）和载荷索引
3. **版本控制**：跟踪点的版本，确保一致性
4. **查询执行**：执行向量搜索和载荷过滤

---

## 五、完整数据流

### 5.1 写入流程

```
用户写入请求
    ↓
Collection::update()
    ↓
Collection::do_core_update()
    ↓
ShardHolder::select_shards()  // 选择目标分片
    ↓
ShardReplicaSet::update()
    ↓
LocalShard::update()
    ↓
LocalShard::update_with_clock()
    ↓
SegmentHolder::get_appendable_segment()  // 获取可追加的段
    ↓
Segment::upsert_point()  // 写入数据
    ↓
WAL::write()  // 写入预写日志
```

### 5.2 搜索流程

```
用户搜索请求
    ↓
Collection::core_search_batch()
    ↓
Collection::do_core_search_batch()
    ↓
ShardHolder::select_shards()  // 选择目标分片
    ↓
ShardReplicaSet::core_search()
    ↓
LocalShard::do_search()
    ↓
SegmentHolder::search()  // 在所有段中搜索
    ↓
Segment::search_batch()  // 每个段执行搜索
    ↓
HNSWIndex::search()  // HNSW 索引搜索
    ↓
合并段的结果
    ↓
合并分片的结果
    ↓
返回给用户
```

---

## 六、存储路径结构

### 6.1 文件系统布局

```
storage/
  └─ collections/
      └─ {collection_name}/          # Collection 目录
          ├─ {shard_id}/              # Shard 目录
          │   ├─ wal/                 # WAL 目录
          │   │   └─ ...
          │   └─ segments/            # Segments 目录
          │       ├─ {segment_id_1}/  # Segment 1 目录
          │       │   ├─ segment.json
          │       │   ├─ vectors.bin
          │       │   ├─ payload.bin
          │       │   ├─ graph.bin     # HNSW 索引
          │       │   └─ links.bin
          │       ├─ {segment_id_2}/  # Segment 2 目录
          │       │   └─ ...
          │       └─ ...
          └─ ...
```

### 6.2 代码中的路径定义

**Shard 路径**：
```41:43:lib/collection/src/shards/mod.rs
pub fn shard_path(collection_path: &Path, shard_id: ShardId) -> PathBuf {
    collection_path.join(format!("{shard_id}"))
}
```

**Segment 路径**：
```477:479:lib/collection/src/shards/local_shard/mod.rs
pub fn segments_path(shard_path: &Path) -> PathBuf {
    shard_path.join(SEGMENTS_PATH)
}
```

---

## 七、数量关系

### 7.1 典型配置

| 组件 | 数量 | 说明 |
|------|------|------|
| **Collection** | 1 | 用户创建的集合 |
| **Shard** | 1-100+ | 根据数据量和节点数决定 |
| **Segment** | 每个 Shard 1-100+ | 根据数据量和优化策略决定 |

### 7.2 示例

```
场景：中等规模的集合

Collection: "my_collection"
  ├─ Shard 0
  │   ├─ Segment 0 (appendable)
  │   ├─ Segment 1
  │   └─ Segment 2
  ├─ Shard 1
  │   ├─ Segment 3 (appendable)
  │   └─ Segment 4
  └─ Shard 2
      ├─ Segment 5 (appendable)
      └─ Segment 6

总计：
  - 1 个 Collection
  - 3 个 Shard
  - 7 个 Segment
```

---

## 八、生命周期管理

### 8.1 Collection 生命周期

```
创建 → 使用 → 删除
  ↓
初始化 ShardHolder
  ↓
创建/加载 Shard
```

### 8.2 Shard 生命周期

```
创建 → 激活 → 使用 → 转移/删除
  ↓
初始化 SegmentHolder
  ↓
创建/加载 Segment
```

### 8.3 Segment 生命周期

```
创建 → 追加数据 → 优化（合并）→ 删除
  ↓
Appendable Segment → Optimized Segment
```

---

## 九、关键设计要点

### 9.1 为什么需要三层结构？

1. **Collection**：用户可见的逻辑单元
   - 提供统一的 API 接口
   - 管理集合级别的配置

2. **Shard**：分布式扩展
   - 支持水平扩展
   - 支持副本和故障转移
   - 可以跨节点分布

3. **Segment**：存储优化
   - 支持增量写入（Appendable Segment）
   - 支持后台优化（合并、删除）
   - 支持索引重建

### 9.2 数据分布策略

#### 9.2.1 Shard 级别

- **自动分片（Auto Sharding）**：数据随机分布到分片
- **自定义分片（Custom Sharding）**：根据 `shard_key` 分布

#### 9.2.2 Segment 级别

- **新数据**：写入 Appendable Segment
- **优化**：后台进程合并多个 Segment
- **删除**：标记删除，优化时物理删除

### 9.3 搜索时的合并

```
Collection 级别：
  - 合并所有 Shard 的结果
  - K-way Merge
  - 去重

Shard 级别：
  - 合并所有 Segment 的结果
  - K-way Merge
  - 去重

Segment 级别：
  - HNSW 索引搜索
  - 返回 Top-K 结果
```

---

## 十、代码位置总结

### 10.1 Collection

| 组件 | 文件 | 说明 |
|------|------|------|
| **Collection 结构** | `lib/collection/src/collection/mod.rs` | Collection 定义 |
| **搜索逻辑** | `lib/collection/src/collection/search.rs` | 搜索协调 |
| **查询逻辑** | `lib/collection/src/collection/query.rs` | 查询协调 |

### 10.2 Shard

| 组件 | 文件 | 说明 |
|------|------|------|
| **Shard 枚举** | `lib/collection/src/shards/shard.rs` | Shard 类型定义 |
| **LocalShard** | `lib/collection/src/shards/local_shard/mod.rs` | 本地分片实现 |
| **ShardHolder** | `lib/collection/src/shards/shard_holder/mod.rs` | 分片管理 |
| **ShardReplicaSet** | `lib/collection/src/shards/replica_set/mod.rs` | 副本集管理 |

### 10.3 Segment

| 组件 | 文件 | 说明 |
|------|------|------|
| **Segment 结构** | `lib/segment/src/segment/mod.rs` | Segment 定义 |
| **SegmentHolder** | `lib/collection/src/collection_manager/holders.rs` | 段管理（通过 shard） |
| **向量索引** | `lib/segment/src/index/` | HNSW 等索引实现 |
| **向量存储** | `lib/segment/src/vector_storage/` | 向量存储实现 |

---

## 十一、总结

### 11.1 关系图

```
Collection (集合)
    │
    ├─ 管理多个 Shard
    │   └─ ShardHolder
    │
    └─ 协调搜索和更新
        └─ 合并结果

Shard (分片)
    │
    ├─ 管理多个 Segment
    │   └─ SegmentHolder
    │
    ├─ 数据持久化
    │   └─ WAL
    │
    └─ 执行搜索
        └─ 合并 Segment 结果

Segment (段)
    │
    ├─ 存储向量数据
    │   └─ VectorStorage
    │
    ├─ 存储载荷数据
    │   └─ PayloadStorage
    │
    ├─ 向量索引
    │   └─ HNSWIndex
    │
    └─ 载荷索引
        └─ PayloadIndex
```

### 11.2 关键特点

1. **层次化组织**：Collection → Shard → Segment
2. **分布式支持**：Shard 可以跨节点分布
3. **存储优化**：Segment 支持增量写入和后台优化
4. **搜索合并**：每个层次都进行结果合并

### 11.3 设计优势

1. **可扩展性**：通过 Shard 实现水平扩展
2. **性能优化**：通过 Segment 优化实现增量写入和后台合并
3. **容错性**：通过 Shard 副本实现故障转移
4. **灵活性**：支持自动分片和自定义分片

---

## 参考资料

- [Qdrant 协调节点实现解读](./Qdrant协调节点实现解读.md)
- [分布式环境下 HNSW 向量搜索工作原理](./分布式环境下HNSW向量搜索工作原理.md)
- [系统架构分析文档](./系统架构分析文档.md)
