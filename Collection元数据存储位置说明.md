# Collection 元数据存储位置说明

## 概述

Qdrant 中 Collection 的元数据主要存储在 Collection 目录下的 JSON 文件中。本文档详细说明各种元数据的存储位置和格式。

---

## 一、存储位置总览

### 1.1 文件系统布局

```
storage/
  └─ collections/
      └─ {collection_name}/          # Collection 目录
          ├─ config.json              # Collection 配置（主要元数据）
          ├─ payload_index.json       # Payload 索引模式
          ├─ {shard_id}/              # Shard 目录
          │   ├─ shard_config.json    # Shard 配置
          │   ├─ wal/                 # WAL 目录
          │   └─ segments/            # Segments 目录
          └─ ...
```

### 1.2 元数据文件列表

| 文件 | 位置 | 说明 |
|------|------|------|
| **config.json** | `{collection_path}/config.json` | Collection 配置（核心元数据） |
| **payload_index.json** | `{collection_path}/payload_index.json` | Payload 索引模式定义 |
| **shard_config.json** | `{collection_path}/{shard_id}/shard_config.json` | Shard 配置（每个 Shard 一个） |

---

## 二、config.json（Collection 配置）

### 2.1 文件位置

**路径**: `{collection_path}/config.json`

**代码位置**: `lib/collection/src/config.rs`

### 2.2 常量定义

```31:31:lib/collection/src/config.rs
pub const COLLECTION_CONFIG_FILE: &str = "config.json";
```

### 2.3 保存方法

```249:257:lib/collection/src/config.rs
    pub fn save(&self, path: &Path) -> CollectionResult<()> {
        let config_path = path.join(COLLECTION_CONFIG_FILE);
        let af = AtomicFile::new(&config_path, AllowOverwrite);
        let state_bytes = serde_json::to_vec(self).unwrap();
        af.write(|f| f.write_all(&state_bytes)).map_err(|err| {
            CollectionError::service_error(format!("Can't write {config_path:?}, error: {err}"))
        })?;
        Ok(())
    }
```

### 2.4 加载方法

```259:265:lib/collection/src/config.rs
    pub fn load(path: &Path) -> CollectionResult<Self> {
        let config_path = path.join(COLLECTION_CONFIG_FILE);
        let mut contents = String::new();
        let mut file = File::open(config_path)?;
        file.read_to_string(&mut contents)?;
        Ok(serde_json::from_str(&contents)?)
    }
```

### 2.5 配置内容

`CollectionConfigInternal` 结构包含以下信息：

```219:242:lib/collection/src/config.rs
#[derive(Debug, Deserialize, Serialize, Validate, Clone, PartialEq)]
pub struct CollectionConfigInternal {
    #[validate(nested)]
    pub params: CollectionParams,
    #[validate(nested)]
    pub hnsw_config: HnswConfig,
    #[validate(nested)]
    pub optimizer_config: OptimizersConfig,
    #[validate(nested)]
    pub wal_config: WalConfig,
    #[serde(default)]
    #[validate(nested)]
    pub quantization_config: Option<QuantizationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(nested)]
    pub strict_mode_config: Option<StrictModeConfig>,
    #[serde(default)]
    pub uuid: Option<Uuid>,
    /// Arbitrary JSON metadata for the collection
    /// This can be used to store application-specific information
    /// such as creation time, migration data, inference model info, etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Payload>,
}
```

### 2.6 配置字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| **params** | `CollectionParams` | 集合参数（向量配置、分片数量、分片方法等） |
| **hnsw_config** | `HnswConfig` | HNSW 索引配置 |
| **optimizer_config** | `OptimizersConfig` | 优化器配置 |
| **wal_config** | `WalConfig` | WAL 配置 |
| **quantization_config** | `Option<QuantizationConfig>` | 量化配置（可选） |
| **strict_mode_config** | `Option<StrictModeConfig>` | 严格模式配置（可选） |
| **uuid** | `Option<Uuid>` | 集合 UUID（可选） |
| **metadata** | `Option<Payload>` | 自定义元数据（可选） |

### 2.7 保存时机

Collection 配置在以下情况下会被保存：

1. **创建 Collection 时**：
```175:177:lib/collection/src/collection/mod.rs
        // Once the config is persisted - the collection is considered to be successfully created.
        CollectionVersion::save(path)?;
        collection_config.save(path)?;
```

2. **更新配置时**（例如更新量化配置）：
```166:166:lib/collection/src/collection/collection_ops.rs
        self.collection_config.read().await.save(&self.path)?;
```

3. **更新元数据时**：
```182:182:lib/collection/src/collection/collection_ops.rs
        self.collection_config.read().await.save(&self.path)?;
```

---

## 三、payload_index.json（Payload 索引模式）

### 3.1 文件位置

**路径**: `{collection_path}/payload_index.json`

**代码位置**: `lib/collection/src/collection/payload_index_schema.rs`

### 3.2 常量定义

```15:15:lib/collection/src/collection/payload_index_schema.rs
pub const PAYLOAD_INDEX_CONFIG_FILE: &str = "payload_index.json";
```

### 3.3 文件路径方法

```18:20:lib/collection/src/collection/payload_index_schema.rs
    pub(crate) fn payload_index_file(collection_path: &Path) -> PathBuf {
        collection_path.join(PAYLOAD_INDEX_CONFIG_FILE)
    }
```

### 3.4 加载方法

```22:29:lib/collection/src/collection/payload_index_schema.rs
    pub(crate) fn load_payload_index_schema(
        collection_path: &Path,
    ) -> CollectionResult<SaveOnDisk<PayloadIndexSchema>> {
        let payload_index_file = Self::payload_index_file(collection_path);
        let schema: SaveOnDisk<PayloadIndexSchema> =
            SaveOnDisk::load_or_init_default(payload_index_file)?;
        Ok(schema)
    }
```

### 3.5 保存机制

`PayloadIndexSchema` 使用 `SaveOnDisk` 包装，自动处理保存：

- 当调用 `write()` 方法修改 schema 时，会自动保存到磁盘
- 使用原子写入确保数据一致性

### 3.6 使用场景

1. **创建 Payload 索引时**：
```50:54:lib/collection/src/collection/payload_index_schema.rs
        self.payload_index_schema.write(|schema| {
            schema
                .schema
                .insert(field_name.clone(), field_schema.clone());
        })?;
```

2. **删除 Payload 索引时**：
```75:77:lib/collection/src/collection/payload_index_schema.rs
        self.payload_index_schema.write(|schema| {
            schema.schema.remove(&field_name);
        })?;
```

---

## 四、Collection State（分布式环境）

### 4.1 概述

在**分布式环境**（集群模式）下，Collection 的状态信息（包括分片分布、副本状态、转移任务等）通过 **Raft 共识机制**存储，而不是直接存储在文件系统中。

### 4.2 State 结构

**代码位置**: `lib/collection/src/collection_state.rs`

```19:30:lib/collection/src/collection_state.rs
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct State {
    pub config: CollectionConfigInternal,
    pub shards: AHashMap<ShardId, ShardInfo>,
    pub resharding: Option<ReshardState>,
    #[serde(default)]
    pub transfers: HashSet<ShardTransfer>,
    #[serde(default)]
    pub shards_key_mapping: ShardKeyMapping,
    #[serde(default)]
    pub payload_index_schema: PayloadIndexSchema,
}
```

### 4.3 State 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| **config** | `CollectionConfigInternal` | Collection 配置（与 config.json 相同） |
| **shards** | `AHashMap<ShardId, ShardInfo>` | 分片信息（每个分片的副本状态） |
| **resharding** | `Option<ReshardState>` | 重分片状态（可选） |
| **transfers** | `HashSet<ShardTransfer>` | 分片转移任务 |
| **shards_key_mapping** | `ShardKeyMapping` | 分片键映射（自定义分片） |
| **payload_index_schema** | `PayloadIndexSchema` | Payload 索引模式（与 payload_index.json 相同） |

### 4.4 存储机制

在分布式环境下：

1. **Raft 共识存储**：Collection State 通过 Raft 共识机制在集群中同步
2. **Raft State 文件**：Raft 共识状态存储在 `storage/raft_state.json`
3. **不直接存储**：Collection State 不直接存储在 Collection 目录下，而是通过 Raft 日志和快照恢复

**代码位置**: `lib/storage/src/content_manager/consensus/persistent.rs`

```26:26:lib/storage/src/content_manager/consensus/persistent.rs
const STATE_FILE_NAME: &str = "raft_state.json";
```

### 4.5 单节点环境

在**单节点环境**下：

- Collection State 的各个组件分别存储：
  - `config` → `config.json`
  - `payload_index_schema` → `payload_index.json`
  - `shards` → 通过 Shard 目录结构隐式存储
  - `transfers` → 不适用（单节点无转移）
  - `resharding` → 不适用（单节点无重分片）

---

## 五、Shard 配置

### 5.1 文件位置

**路径**: `{collection_path}/{shard_id}/shard_config.json`

**代码位置**: `lib/collection/src/shards/shard_config.rs`

### 5.2 常量定义

```10:10:lib/collection/src/shards/shard_config.rs
pub const SHARD_CONFIG_FILE: &str = "shard_config.json";
```

### 5.3 说明

每个 Shard 都有自己的配置文件，存储在 Shard 目录下。这个文件包含 Shard 特定的配置信息。

---

## 六、元数据更新流程

### 6.1 配置更新流程

```
用户更新请求
    ↓
Collection::update_xxx_config()
    ↓
修改内存中的配置
    ↓
CollectionConfigInternal::save()
    ↓
原子写入 config.json
```

### 6.2 Payload 索引更新流程

```
用户创建/删除索引
    ↓
Collection::create_payload_index() / drop_payload_index()
    ↓
PayloadIndexSchema::write()
    ↓
SaveOnDisk 自动保存到 payload_index.json
```

### 6.3 分布式环境下的状态更新

```
用户操作（创建/更新/删除 Collection）
    ↓
通过 Raft 共识提交操作
    ↓
Raft 日志记录
    ↓
应用到所有节点
    ↓
更新本地 config.json 和 payload_index.json
```

---

## 七、文件格式示例

### 7.1 config.json 示例

```json
{
  "params": {
    "vectors": {
      "size": 128,
      "distance": "Cosine"
    },
    "shard_number": 1,
    "sharding_method": "auto",
    "replication_factor": 1,
    "write_consistency_factor": 1,
    "on_disk_payload": true
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 10000
  },
  "optimizer_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000,
    "default_segment_number": 0,
    "max_segment_size": null,
    "memmap_threshold": null,
    "indexing_threshold": 20000,
    "flush_interval_sec": 5,
    "max_optimization_threads": 2
  },
  "wal_config": {
    "wal_capacity_mb": 32,
    "wal_segments_ahead": 0,
    "wal_retain_closed": 1
  },
  "quantization_config": null,
  "strict_mode_config": null,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "metadata": {
    "created_at": "2024-01-01T00:00:00Z",
    "description": "My collection"
  }
}
```

### 7.2 payload_index.json 示例

```json
{
  "schema": {
    "category": {
      "data_type": "keyword",
      "indexed": true
    },
    "price": {
      "data_type": "float",
      "indexed": true
    },
    "location": {
      "data_type": "geo",
      "indexed": true
    }
  }
}
```

---

## 八、关键设计要点

### 8.1 原子写入

所有元数据文件都使用**原子写入**（`AtomicFile`）确保：
- 写入过程中不会产生损坏的文件
- 写入失败时不会丢失原有数据
- 并发安全

### 8.2 单节点 vs 分布式

| 环境 | 存储方式 | 说明 |
|------|---------|------|
| **单节点** | 直接文件存储 | `config.json` 和 `payload_index.json` 直接存储在 Collection 目录 |
| **分布式** | Raft 共识 + 文件存储 | State 通过 Raft 同步，但 `config.json` 和 `payload_index.json` 仍然存储在本地 |

### 8.3 元数据一致性

1. **配置一致性**：`config.json` 是配置的权威来源
2. **索引模式一致性**：`payload_index.json` 是索引模式的权威来源
3. **分布式一致性**：通过 Raft 共识确保集群中所有节点的元数据一致

### 8.4 恢复机制

1. **启动时加载**：Qdrant 启动时会从 `config.json` 和 `payload_index.json` 加载元数据
2. **Raft 恢复**：分布式环境下，通过 Raft 日志和快照恢复 Collection State
3. **版本兼容性**：支持版本迁移和兼容性检查

---

## 九、代码位置总结

### 9.1 Collection 配置

| 组件 | 文件 | 说明 |
|------|------|------|
| **配置结构** | `lib/collection/src/config.rs` | `CollectionConfigInternal` 定义 |
| **保存/加载** | `lib/collection/src/config.rs` | `save()` 和 `load()` 方法 |
| **更新操作** | `lib/collection/src/collection/collection_ops.rs` | 各种配置更新方法 |

### 9.2 Payload 索引模式

| 组件 | 文件 | 说明 |
|------|------|------|
| **Schema 结构** | `lib/collection/src/collection/payload_index_schema.rs` | `PayloadIndexSchema` 定义 |
| **保存/加载** | `lib/collection/src/collection/payload_index_schema.rs` | 使用 `SaveOnDisk` 包装 |
| **索引操作** | `lib/collection/src/collection/payload_index_schema.rs` | `create_payload_index()` 和 `drop_payload_index()` |

### 9.3 Collection State

| 组件 | 文件 | 说明 |
|------|------|------|
| **State 结构** | `lib/collection/src/collection_state.rs` | `State` 定义 |
| **状态管理** | `lib/collection/src/collection/state_management.rs` | `apply_state()` 等方法 |
| **Raft 存储** | `lib/storage/src/content_manager/consensus/persistent.rs` | Raft 状态持久化 |

---

## 十、总结

### 10.1 存储位置总结

1. **config.json**：Collection 目录下，包含所有配置信息
2. **payload_index.json**：Collection 目录下，包含 Payload 索引模式
3. **shard_config.json**：每个 Shard 目录下，包含 Shard 配置
4. **raft_state.json**：存储根目录下，包含 Raft 共识状态（分布式环境）

### 10.2 关键特点

1. **原子写入**：所有元数据文件使用原子写入确保数据安全
2. **自动保存**：配置更新时自动保存到磁盘
3. **版本兼容**：支持版本迁移和兼容性检查
4. **分布式同步**：分布式环境下通过 Raft 共识同步状态

### 10.3 最佳实践

1. **备份元数据**：定期备份 `config.json` 和 `payload_index.json`
2. **监控文件**：监控元数据文件的大小和修改时间
3. **版本控制**：了解 Qdrant 版本升级对元数据格式的影响
4. **恢复测试**：定期测试从元数据文件恢复 Collection 的能力

---

## 参考资料

- [Collection、Shard、Segment 关系详解](./Collection_Shard_Segment关系详解.md)
- [Qdrant 协调节点实现解读](./Qdrant协调节点实现解读.md)
- [系统架构分析文档](./系统架构分析文档.md)
