# Qdrant 中 RocksDB 的角色说明

## 概述

RocksDB 在 Qdrant 中充当**可选的持久化存储引擎**，主要用于存储向量数据和 payload 数据。**注意：Qdrant 正在逐步移除对 RocksDB 的依赖**，计划迁移到基于内存映射文件（mmap）的存储方案。

## RocksDB 的主要角色

### 1. **持久化存储引擎**

RocksDB 作为底层键值存储引擎，为 Qdrant 提供持久化存储能力。

**位置**: `lib/segment/src/common/rocksdb_wrapper.rs`

**配置选项**:
```rust
const DB_CACHE_SIZE: usize = 10 * 1024 * 1024; // 10 MB
const DB_MAX_LOG_SIZE: usize = 1024 * 1024; // 1 MB
const DB_MAX_OPEN_FILES: usize = 256;
const DB_DELETE_OBSOLETE_FILES_PERIOD: u64 = 3 * 60 * 1_000_000; // 3 分钟

// 压缩类型
options.set_compression_type(rocksdb::DBCompressionType::Lz4);

// Qdrant 依赖自己的 WAL 来保证持久性
options.set_wal_recovery_mode(DBRecoveryMode::TolerateCorruptedTailRecords);
```

### 2. **列族（Column Family）存储**

RocksDB 使用列族来组织不同类型的数据：

```rust
pub const DB_VECTOR_CF: &str = "vector";      // 向量数据
pub const DB_PAYLOAD_CF: &str = "payload";   // Payload 数据
pub const DB_MAPPING_CF: &str = "mapping";    // 映射关系（已废弃）
pub const DB_VERSIONS_CF: &str = "version";   // 版本信息（已废弃）
pub const DB_DEFAULT_CF: &str = "default";    // 默认列族
```

#### 2.1 向量数据存储

**位置**: `lib/segment/src/vector_storage/`

当启用 `rocksdb` feature 时，可以使用以下向量存储类型：

- **`DenseSimple`**: 密集向量存储（f32）
- **`DenseSimpleByte`**: 密集向量存储（u8）
- **`DenseSimpleHalf`**: 密集向量存储（f16）
- **`MultiDenseSimple`**: 多向量存储

**特点**:
- 向量数据存储在 RocksDB 的列族中
- 每个向量名称对应一个列族（如 `vector:my_vector`）
- 支持稀疏向量的磁盘存储（`SparseVectorStorageType::OnDisk`）

#### 2.2 Payload 数据存储

**位置**: `lib/segment/src/payload_storage/on_disk_payload_storage.rs`

RocksDB 支持两种 payload 存储模式：

1. **`OnDiskPayloadStorage`**: 
   - 所有 payload 数据存储在磁盘上
   - 每次请求时从磁盘读取
   - 不占用内存，但读取速度较慢

2. **`SimplePayloadStorage`** (InMemory):
   - Payload 数据存储在内存中
   - 仅在向量更改时持久化到 RocksDB
   - 读取速度快，但占用内存

**实现**:
```rust
pub struct OnDiskPayloadStorage {
    db_wrapper: DatabaseColumnScheduledDeleteWrapper,
}

impl OnDiskPayloadStorage {
    pub fn open(database: Arc<RwLock<DB>>) -> OperationResult<Self> {
        let db_wrapper = DatabaseColumnScheduledDeleteWrapper::new(
            DatabaseColumnWrapper::new(database, DB_PAYLOAD_CF)
        );
        Ok(OnDiskPayloadStorage { db_wrapper })
    }
}
```

### 3. **可选特性（Feature Flag）**

RocksDB 是一个**可选特性**，通过 `rocksdb` feature flag 控制：

**位置**: `Cargo.toml`

```toml
rocksdb = ["collection/rocksdb", "segment/rocksdb"]
```

**影响**:
- 如果未启用 `rocksdb` feature，Qdrant 将使用其他存储方案（如 mmap）
- 新创建的集合默认不使用 RocksDB（如果 feature flags 启用）

## 正在被移除

### 1. **迁移计划**

根据代码注释和路线图，Qdrant **计划完全移除对 RocksDB 的依赖**：

**证据**:

1. **代码注释** (`lib/segment/src/segment_constructor/rocksdb_builder.rs`):
   ```rust
   /// Struct to optionally create and open a RocksDB instance in a lazy way.
   /// Used as helper to eventually completely remove RocksDB.
   ```

2. **路线图** (`docs/roadmap/README.md`):
   > We plan to eventually migrate away from RocksDB, so at some point we might introduce data migration into the upgrade process.

3. **Feature Flags** (`lib/common/common/src/flags.rs`):
   - `payload_index_skip_rocksdb`: 跳过在新不可变 payload 索引中使用 RocksDB
   - `payload_index_skip_mutable_rocksdb`: 跳过在新可变 payload 索引中使用 RocksDB
   - `payload_storage_skip_rocksdb`: 跳过在新 payload 存储中使用 RocksDB
   - `migrate_rocksdb_*`: 迁移现有 RocksDB 数据到新格式

### 2. **迁移功能**

Qdrant 提供了自动迁移功能，将 RocksDB 数据迁移到新的存储格式：

**位置**: `lib/segment/src/segment_constructor/segment_constructor_base.rs`

```rust
/// Migrate a RocksDB based payload storage into the mmap format
/// Creates a new payload storage on top of memory maps, and copies all payloads
/// from the RocksDB based storage into it.
pub fn migrate_rocksdb_payload_storage_to_mmap(...) -> OperationResult<MmapPayloadStorage>
```

**迁移内容**:
- Payload 存储：从 RocksDB 迁移到 Mmap
- 向量存储：从 RocksDB 迁移到文件系统存储
- ID 跟踪器：从 RocksDB 迁移到基于文件的 ID 跟踪器

### 3. **替代方案**

Qdrant 正在迁移到以下替代方案：

1. **Mmap（内存映射文件）**:
   - 用于 payload 存储（`MmapPayloadStorage`）
   - 性能更好，内存使用更灵活
   - 支持 `Mmap` 和 `InRamMmap` 两种模式

2. **文件系统存储**:
   - 用于向量存储（`DenseMemmap`, `DenseAppendableMemmap` 等）
   - 直接使用内存映射文件，无需 RocksDB

3. **Gridstore**:
   - 用于某些场景的 blob 存储
   - 在基准测试中与 RocksDB 对比

## 使用场景

### 当前使用场景

1. **向后兼容**:
   - 支持旧版本 Qdrant 创建的集合
   - 自动迁移到新格式

2. **可选存储选项**:
   - 如果启用了 `rocksdb` feature，仍可使用 RocksDB 存储
   - 主要用于兼容性目的

### 不再推荐使用

- **新集合**: 默认不使用 RocksDB（如果 feature flags 启用）
- **新部署**: 建议使用 mmap 或文件系统存储

## 配置和启用

### 启用 RocksDB Feature

在构建时启用 `rocksdb` feature：

```bash
cargo build --features rocksdb
```

### 配置选项

RocksDB 的配置选项在 `lib/segment/src/common/rocksdb_wrapper.rs` 中定义：

- **缓存大小**: 10 MB（每个列族）
- **日志大小**: 1 MB
- **最大打开文件数**: 256
- **压缩类型**: LZ4
- **WAL 恢复模式**: `TolerateCorruptedTailRecords`（因为 Qdrant 使用自己的 WAL）

## 总结

### RocksDB 的角色

1. **历史遗留**: 作为 Qdrant 早期版本的存储引擎
2. **可选特性**: 通过 feature flag 控制，不是必需的
3. **正在被移除**: Qdrant 计划完全移除对 RocksDB 的依赖
4. **迁移支持**: 提供自动迁移功能，将 RocksDB 数据迁移到新格式

### 主要用途

- **向量数据存储**: 通过列族存储向量数据
- **Payload 存储**: 支持内存和磁盘两种模式
- **向后兼容**: 支持旧版本数据格式

### 未来方向

- **完全移除**: 计划在未来版本中完全移除 RocksDB
- **迁移到 Mmap**: 使用内存映射文件作为主要存储方案
- **性能优化**: 新存储方案提供更好的性能和更灵活的内存管理

## 相关文件

- **RocksDB 包装器**: `lib/segment/src/common/rocksdb_wrapper.rs`
- **Payload 存储**: `lib/segment/src/payload_storage/on_disk_payload_storage.rs`
- **向量存储**: `lib/segment/src/vector_storage/dense/simple_dense_vector_storage.rs`
- **迁移功能**: `lib/segment/src/segment_constructor/segment_constructor_base.rs`
- **Feature Flags**: `lib/common/common/src/flags.rs`
- **RocksDB Builder**: `lib/segment/src/segment_constructor/rocksdb_builder.rs`
