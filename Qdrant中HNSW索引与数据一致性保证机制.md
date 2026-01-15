# Qdrant 中 HNSW 索引与数据一致性保证机制

## 概述

Qdrant 中 HNSW 索引存储在独立文件中，而向量数据可能存储在 RocksDB 或 mmap 文件中。为了确保索引与数据的一致性，Qdrant 采用了多层次的机制：

1. **删除标记（Deleted Bitslice）**：在向量存储中维护删除标记
2. **ID Tracker**：维护外部 ID 和内部 ID 的映射关系
3. **序列号（Sequence Number）**：跟踪操作的顺序和版本
4. **WAL（Write-Ahead Log）**：保证操作的持久性和可恢复性
5. **Flush 顺序**：确保数据持久化的顺序一致性
6. **版本控制**：防止重复操作和保证操作顺序
7. **延迟索引更新**：通过后台优化器定期重建索引

---

## 1. 删除标记（Deleted Bitslice）

### 1.1 机制说明

HNSW 索引**不会立即删除**图中的节点，而是通过**删除标记**来标记已删除的点。在搜索时，会过滤掉已删除的点。

### 1.2 实现细节

```rust
// 获取删除标记位图
let deleted_bitslice = vector_storage.deleted_vector_bitslice();

// 在搜索时过滤已删除的点
payload_index
    .iter_filtered_points(
        &filter,
        id_tracker,
        &cardinality_estimation,
        &disposed_hw_counter,
        stopped,
    )
    .filter(|&point_id| !deleted_bitslice.get_bit(point_id as usize).unwrap_or(false))
    .collect()
```

**位置**: `lib/segment/src/index/hnsw_index/hnsw.rs:726-740`

### 1.3 工作原理

1. **向量存储维护删除标记**：
   - 每个点对应一个位（bit）
   - `true` 表示已删除，`false` 表示存在

2. **HNSW 索引使用删除标记**：
   - 在构建索引时，排除已删除的点
   - 在搜索时，过滤掉已删除的点
   - 索引中的节点不会被物理删除

3. **优势**：
   - 避免频繁修改图结构
   - 保持索引的稳定性
   - 提高搜索性能

---

## 2. ID Tracker（ID 跟踪器）

### 2.1 机制说明

ID Tracker 维护外部 ID（用户可见的 ID）和内部 ID（向量存储中的偏移量）之间的映射关系，以及每个点的版本号。

### 2.2 数据结构

```rust
pub struct MergedPointId {
    /// Unique external ID. If the same external ID is present in multiple
    /// trackers, the item with the highest version takes precedence.
    pub external_id: ExtendedPointId,
    /// An index within `id_trackers` iterator that points to the [`IdTracker`]
    /// that contains this point.
    pub tracker_index: usize,
    /// The internal ID of the point within the [`IdTracker`] that contains it.
    pub internal_id: PointOffsetType,
    /// The version of the point within the [`IdTracker`] that contains it.
    pub version: u64,
}
```

**位置**: `lib/segment/src/id_tracker/mod.rs:18-30`

### 2.3 作用

1. **ID 映射**：
   - 外部 ID（如 `100`, `"uuid-123"`）→ 内部 ID（如 `0`, `1`, `2`）
   - 内部 ID 是向量存储中的偏移量

2. **版本跟踪**：
   - 每个点都有一个版本号
   - 版本号用于检测重复操作和保证操作顺序

3. **一致性保证**：
   - 确保索引中的点 ID 与向量存储中的点 ID 一致
   - 通过内部 ID 关联索引和向量数据

---

## 3. 序列号（Sequence Number）

### 3.1 机制说明

每个操作都有一个唯一的序列号（`SeqNumberType`），用于：
- 跟踪操作的顺序
- 检测重复操作
- 保证操作的幂等性

### 3.2 实现细节

```rust
/// Manage point version checking inside this segment, for point level operations
///
/// If current version is higher than operation version - do not perform the operation
/// Update current version if operation successfully executed
pub(super) fn handle_point_version<F>(
    &mut self,
    op_num: SeqNumberType,
    op_point_offset: Option<PointOffsetType>,
    operation: F,
) -> OperationResult<bool>
where
    F: FnOnce(&mut Segment) -> OperationResult<(bool, Option<PointOffsetType>)>,
{
    // If point does not exist or has lower version, ignore operation
    if let Some(point_offset) = op_point_offset
        && self
            .id_tracker
            .borrow()
            .internal_version(point_offset)
            .is_some_and(|current_version| current_version > op_num)
    {
        return Ok(false);
    }

    let (applied, internal_id) = operation(self)?;

    self.bump_segment_version(op_num);
    if let Some(internal_id) = internal_id {
        self.id_tracker
            .borrow_mut()
            .set_internal_version(internal_id, op_num)?;
    }

    Ok(applied)
}
```

**位置**: `lib/segment/src/segment/segment_ops.rs:269-299`

### 3.3 工作原理

1. **操作顺序保证**：
   - 如果操作的序列号小于当前版本，操作会被忽略
   - 确保操作按顺序执行

2. **幂等性**：
   - 相同序列号的操作只会执行一次
   - 防止重复操作导致的不一致

3. **版本更新**：
   - 操作成功后，更新点和 segment 的版本号
   - 版本号用于检测操作是否已应用

---

## 4. WAL（Write-Ahead Log）

### 4.1 机制说明

WAL 是 Qdrant 的核心持久化机制，所有操作都先写入 WAL，然后才应用到内存中的数据结构。

### 4.2 实现细节

```rust
/// Write a record to the WAL, guarantee durability.
///
/// On success, this returns the WAL record number of the written operation along with a WAL
/// lock guard.
#[must_use = "returned record number and WAL lock must be used carefully"]
pub async fn lock_and_write(
    &self,
    operation: &mut OperationWithClockTag,
) -> shard::wal::Result<(u64, OwnedMutexGuard<SerdeWal<OperationWithClockTag>>)> {
    // Update last seen clock map and correct clock tag if necessary
    if let Some(clock_tag) = &mut operation.clock_tag {
        let operation_accepted = self
            .newest_clocks
            .lock()
            .await
            .advance_clock_and_correct_tag(clock_tag);

        if !operation_accepted {
            return Err(shard::wal::WalError::ClockRejected);
        }
    }

    // Write operation to WAL
    let mut wal_lock = Mutex::lock_owned(self.wal.clone()).await;
    wal_lock.write(operation).map(|op_num| (op_num, wal_lock))
}
```

**位置**: `lib/collection/src/wal_delta.rs:50-70`

### 4.3 作用

1. **持久性保证**：
   - 所有操作先写入 WAL
   - 即使系统崩溃，也可以从 WAL 恢复

2. **一致性恢复**：
   - 系统重启时，从 WAL 重放操作
   - 确保索引和数据的一致性

3. **操作顺序**：
   - WAL 中的操作按顺序存储
   - 重放时按顺序执行，保证一致性

---

## 5. Flush 顺序

### 5.1 机制说明

Qdrant 有明确的 flush 顺序，确保数据持久化的顺序一致性。

### 5.2 实现细节

```rust
// Flush order is important:
//
// 1. Flush id mapping. So during recovery the point will be recovered in proper segment.
// 2. Flush vectors and payloads.
// 3. Flush id versions last. So presence of version indicates that all other data is up-to-date.
//
// Example of recovery from WAL in case of partial flush:
//
// In-memory state:
//
//     Segment 1                  Segment 2
//
//    ID-mapping     vst.1       ID-mapping     vst.2
//   ext     int
//  ┌───┐   ┌───┐   ┌───┐       ┌───┐   ┌───┐   ┌───┐
//  │100├───┤1  │   │1  │       │300├───┤1  │   │1  │
//  └───┘   └───┘   │2  │       └───┘   └───┘   │2  │
//                  │   │                       │   │
//  ┌───┐   ┌───┐   │   │       ┌───┐   ┌───┐   │   │
//  │200├───┤2  │   │   │       │400├───┤2  │   │   │
//  └───┘   └───┘   └───┘       └───┘   └───┘   └───┘
```

**位置**: `lib/segment/src/segment/entry.rs:657-703`

### 5.3 Flush 顺序

1. **第一步：Flush ID 映射**
   - 确保点的外部 ID 和内部 ID 的映射关系已持久化
   - 恢复时，点可以被正确分配到对应的 segment

2. **第二步：Flush 向量和 Payload**
   - 持久化向量数据和 payload 数据
   - 确保数据已写入磁盘

3. **第三步：Flush ID 版本**
   - 最后持久化版本信息
   - 版本信息的存在表示所有其他数据都是最新的

### 5.4 恢复机制

如果 flush 过程中系统崩溃，恢复时会：
1. 从 WAL 读取未持久化的操作
2. 根据版本号判断哪些操作需要重放
3. 按顺序重放操作，恢复一致性

---

## 6. 版本控制

### 6.1 Segment 版本

每个 segment 都有版本号，用于跟踪最新的操作：

```rust
pub struct Segment {
    /// Initial version this segment was created at
    pub initial_version: Option<SeqNumberType>,
    /// Latest update operation number, applied to this segment
    /// If None, there were no updates and segment is empty
    pub version: Option<SeqNumberType>,
    /// Latest persisted version
    /// Locked structure on which we hold the lock during flush to prevent concurrent flushes
    pub persisted_version: Arc<Mutex<Option<SeqNumberType>>>,
    // ...
}
```

**位置**: `lib/segment/src/segment/mod.rs:64-93`

### 6.2 点版本

每个点也有版本号，用于检测重复操作：

```rust
pub fn point_version(&self, point_id: PointIdType) -> Option<SeqNumberType> {
    self.id_tracker
        .borrow()
        .external_to_internal(point_id)
        .and_then(|internal_id| id_tracker.internal_version(internal_id))
}
```

**位置**: `lib/segment/src/segment/entry.rs:51-55`

### 6.3 版本跟踪器

`VersionTracker` 跟踪不同子结构的版本：

```rust
pub struct VersionTracker {
    /// Tracks version of *mutable* files inside vector storage.
    /// Should be updated when vector storage is modified.
    vector_storage: HashMap<VectorNameBuf, SeqNumberType>,

    /// Tracks version of *mutable* files inside payload storage.
    /// Should be updated when payload storage is modified.
    payload_storage: Option<SeqNumberType>,

    /// Tracks version of *immutable* files inside payload index.
    /// Should be updated when payload index *schema* of the field is modified.
    payload_index_schema: HashMap<JsonPath, SeqNumberType>,
}
```

**位置**: `lib/segment/src/segment/version_tracker.rs:10-30`

---

## 7. 延迟索引更新

### 7.1 机制说明

HNSW 索引**不是实时更新**的，而是通过后台优化器定期重建。

### 7.2 工作原理

1. **操作时**：
   - 向量数据立即更新
   - 删除标记立即更新
   - HNSW 索引**不立即更新**

2. **搜索时**：
   - 使用现有的 HNSW 索引进行搜索
   - 通过删除标记过滤已删除的点
   - 通过 ID Tracker 验证点的存在性

3. **后台优化**：
   - 优化器定期检查 segment
   - 如果删除的点太多或索引过时，触发索引重建
   - 重建时，排除已删除的点，只索引存在的点

### 7.3 优势

1. **性能**：
   - 避免频繁修改图结构
   - 保持索引的稳定性
   - 提高搜索性能

2. **一致性**：
   - 通过删除标记保证搜索结果的正确性
   - 通过 ID Tracker 验证点的存在性
   - 定期重建确保索引的准确性

---

## 8. 一致性保证流程

### 8.1 插入操作

```
1. 操作写入 WAL
   ↓
2. 分配内部 ID（ID Tracker）
   ↓
3. 存储向量数据（向量存储）
   ↓
4. 更新删除标记（标记为未删除）
   ↓
5. 更新版本号（点和 segment）
   ↓
6. 后台优化器定期重建索引（包含新点）
```

### 8.2 删除操作

```
1. 操作写入 WAL
   ↓
2. 通过 ID Tracker 查找内部 ID
   ↓
3. 更新删除标记（标记为已删除）
   ↓
4. 更新版本号（点和 segment）
   ↓
5. 搜索时过滤已删除的点
   ↓
6. 后台优化器定期重建索引（排除已删除的点）
```

### 8.3 更新操作

```
1. 操作写入 WAL
   ↓
2. 通过 ID Tracker 查找内部 ID
   ↓
3. 检查版本号（防止旧操作覆盖新数据）
   ↓
4. 更新向量数据（向量存储）
   ↓
5. 更新删除标记（如果之前被删除，现在恢复）
   ↓
6. 更新版本号（点和 segment）
   ↓
7. 后台优化器定期重建索引（包含更新的点）
```

### 8.4 恢复流程

```
1. 系统启动
   ↓
2. 加载持久化的数据（向量、payload、ID 映射）
   ↓
3. 读取 WAL 中的操作
   ↓
4. 比较操作版本号和持久化版本号
   ↓
5. 重放未持久化的操作
   ↓
6. 重建索引（如果需要）
```

---

## 9. 总结

Qdrant 通过以下机制保证 HNSW 索引与数据的一致性：

1. **删除标记**：在向量存储中维护删除标记，搜索时过滤已删除的点
2. **ID Tracker**：维护外部 ID 和内部 ID 的映射关系，以及每个点的版本号
3. **序列号**：跟踪操作的顺序，防止重复操作
4. **WAL**：保证操作的持久性和可恢复性
5. **Flush 顺序**：确保数据持久化的顺序一致性
6. **版本控制**：防止重复操作和保证操作顺序
7. **延迟索引更新**：通过后台优化器定期重建索引，避免频繁修改图结构

这些机制共同工作，确保了即使 HNSW 索引存储在独立文件中，也能与向量数据保持一致性。

---

## 10. 代码位置总结

### 10.1 删除标记

- **向量存储删除标记**: `lib/segment/src/vector_storage/` - `deleted_vector_bitslice()`
- **HNSW 索引使用删除标记**: `lib/segment/src/index/hnsw_index/hnsw.rs:726-740`

### 10.2 ID Tracker

- **ID Tracker 实现**: `lib/segment/src/id_tracker/`
- **版本跟踪**: `lib/segment/src/id_tracker/compressed/versions_store.rs`

### 10.3 序列号和版本控制

- **Segment 版本**: `lib/segment/src/segment/mod.rs:64-93`
- **点版本处理**: `lib/segment/src/segment/segment_ops.rs:269-299`
- **版本跟踪器**: `lib/segment/src/segment/version_tracker.rs`

### 10.4 WAL

- **WAL 实现**: `lib/shard/src/wal.rs`
- **WAL 写入**: `lib/collection/src/wal_delta.rs:50-70`

### 10.5 Flush 顺序

- **Flush 实现**: `lib/segment/src/segment/entry.rs:622-750`
