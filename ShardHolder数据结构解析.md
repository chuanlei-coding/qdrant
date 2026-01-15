# ShardHolder 数据结构解析

## 概述

`ShardHolder` 是 Qdrant 中管理 Collection 所有分片（Shard）的核心数据结构。它负责分片的存储、路由、转移和重分片等操作。

---

## 一、数据结构定义

### 1.1 核心结构

**文件位置**: `lib/collection/src/shards/shard_holder/mod.rs`

```61:75:lib/collection/src/shards/shard_holder/mod.rs
pub struct ShardHolder {
    shards: AHashMap<ShardId, ShardReplicaSet>,
    pub(crate) shard_transfers: SaveOnDisk<HashSet<ShardTransfer>>,
    pub(crate) shard_transfer_changes: broadcast::Sender<ShardTransferChange>,
    pub(crate) resharding_state: SaveOnDisk<Option<ReshardState>>,
    /// Hash rings per shard key
    ///
    /// In case of auto sharding, this only hash a `None` hash ring. In case of custom sharding,
    /// this only has hash rings for defined shard keys excluding `None`.
    pub(crate) rings: HashMap<Option<ShardKey>, HashRingRouter>,
    key_mapping: SaveOnDisk<ShardKeyMapping>,
    // Duplicates the information from `key_mapping` for faster access, does not use locking
    shard_id_to_key_mapping: AHashMap<ShardId, ShardKey>,
    sharding_method: ShardingMethod,
}
```

### 1.2 类型别名

```77:77:lib/collection/src/shards/shard_holder/mod.rs
pub type LockedShardHolder = RwLock<ShardHolder>;
```

`LockedShardHolder` 是 `ShardHolder` 的读写锁包装，用于并发访问控制。

---

## 二、字段详解

### 2.1 shards: AHashMap<ShardId, ShardReplicaSet>

**类型**: `AHashMap<ShardId, ShardReplicaSet>`

**说明**: 存储所有分片的映射表，键为分片 ID，值为分片副本集。

**特点**:
- 使用 `AHashMap`（ahash 库）提供高性能的哈希映射
- `ShardId` 是 `u32` 类型
- `ShardReplicaSet` 管理一个分片的所有副本（本地和远程）

**相关方法**:
```357:363:lib/collection/src/shards/shard_holder/mod.rs
    pub fn get_shard(&self, shard_id: ShardId) -> Option<&ShardReplicaSet> {
        self.shards.get(&shard_id)
    }

    pub fn get_shard_mut(&mut self, shard_id: ShardId) -> Option<&mut ShardReplicaSet> {
        self.shards.get_mut(&shard_id)
    }
```

### 2.2 shard_transfers: SaveOnDisk<HashSet<ShardTransfer>>

**类型**: `SaveOnDisk<HashSet<ShardTransfer>>`

**说明**: 持久化存储的分片转移任务集合。

**特点**:
- 使用 `SaveOnDisk` 包装，自动持久化到磁盘
- 存储文件: `{collection_path}/shard_transfers`
- 记录所有进行中的分片转移任务

**相关常量**:
```57:57:lib/collection/src/shards/shard_holder/mod.rs
const SHARD_TRANSFERS_FILE: &str = "shard_transfers";
```

**相关方法**:
```428:436:lib/collection/src/shards/shard_holder/mod.rs
    pub fn register_start_shard_transfer(&self, transfer: ShardTransfer) -> CollectionResult<bool> {
        let changed = self
            .shard_transfers
            .write(|transfers| transfers.insert(transfer.clone()))?;
        let _ = self
            .shard_transfer_changes
            .send(ShardTransferChange::Start(transfer));
        Ok(changed)
    }
```

### 2.3 shard_transfer_changes: broadcast::Sender<ShardTransferChange>

**类型**: `broadcast::Sender<ShardTransferChange>`

**说明**: 分片转移状态变更的广播通道发送端。

**特点**:
- 使用 `tokio::sync::broadcast` 实现多订阅者模式
- 通道容量: 64（初始化时设置）
- 用于通知分片转移的开始、完成或中止

**ShardTransferChange 枚举**:
```1512:1516:lib/collection/src/shards/shard_holder/mod.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShardTransferChange {
    Start(ShardTransfer),
    Finish(ShardTransferKey),
    Abort(ShardTransferKey),
}
```

**初始化**:
```111:111:lib/collection/src/shards/shard_holder/mod.rs
        let (shard_transfer_changes, _) = broadcast::channel(64);
```

### 2.4 resharding_state: SaveOnDisk<Option<ReshardState>>

**类型**: `SaveOnDisk<Option<ReshardState>>`

**说明**: 持久化存储的重分片状态。

**特点**:
- 使用 `SaveOnDisk` 包装，自动持久化到磁盘
- 存储文件: `{collection_path}/resharding_state.json`
- `None` 表示没有进行中的重分片操作

**相关常量**:
```58:58:lib/collection/src/shards/shard_holder/mod.rs
const RESHARDING_STATE_FILE: &str = "resharding_state.json";
```

### 2.5 rings: HashMap<Option<ShardKey>, HashRingRouter>

**类型**: `HashMap<Option<ShardKey>, HashRingRouter>`

**说明**: 每个分片键对应的哈希环路由器。

**特点**:
- **Auto Sharding**: 只有一个 `None` 键的哈希环
- **Custom Sharding**: 每个分片键都有独立的哈希环（不包括 `None`）
- `HashRingRouter` 用于将点路由到相应的分片

**初始化**:
```106:109:lib/collection/src/shards/shard_holder/mod.rs
        let rings = match sharding_method {
            ShardingMethod::Auto => HashMap::from([(None, HashRingRouter::single())]),
            ShardingMethod::Custom => HashMap::new(),
        };
```

**HashRingRouter 结构**:
```17:25:lib/collection/src/hash_ring.rs
#[derive(Clone, Debug, PartialEq)]
pub enum HashRingRouter<T: Eq + StableHash + Hash = ShardId> {
    /// Single hashring
    Single(HashRing<T>),

    /// Two hashrings when transitioning during resharding
    /// Depending on the current resharding state, points may be in either or both shards.
    Resharding { old: HashRing<T>, new: HashRing<T> },
}
```

### 2.6 key_mapping: SaveOnDisk<ShardKeyMapping>

**类型**: `SaveOnDisk<ShardKeyMapping>`

**说明**: 持久化存储的分片键到分片 ID 的映射。

**特点**:
- 使用 `SaveOnDisk` 包装，自动持久化到磁盘
- 存储文件: `{collection_path}/shard_key_mapping.json`
- 用于自定义分片（Custom Sharding）模式

**相关常量**:
```59:59:lib/collection/src/shards/shard_holder/mod.rs
pub const SHARD_KEY_MAPPING_FILE: &str = "shard_key_mapping.json";
```

**ShardKeyMapping 结构**:
```14:21:lib/collection/src/shards/shard_holder/shard_mapping.rs
pub struct ShardKeyMapping {
    shard_key_to_shard_ids: HashMap<ShardKey, HashSet<ShardId>>,

    /// `true` if the ShardKeyMapping was specified in the old format.
    // TODO(1.17.0): Remove once all keys are migrated.
    #[serde(skip)]
    pub(crate) was_old_format: bool,
}
```

### 2.7 shard_id_to_key_mapping: AHashMap<ShardId, ShardKey>

**类型**: `AHashMap<ShardId, ShardKey>`

**说明**: 分片 ID 到分片键的反向映射（用于快速查找）。

**特点**:
- 从 `key_mapping` 中提取的反向映射
- 不使用锁，提供更快的访问速度
- 在 `key_mapping` 更新时同步更新

**初始化**:
```98:104:lib/collection/src/shards/shard_holder/mod.rs
        let mut shard_id_to_key_mapping = AHashMap::new();

        for (shard_key, shard_ids) in key_mapping.read().iter() {
            for shard_id in shard_ids {
                shard_id_to_key_mapping.insert(*shard_id, shard_key.clone());
            }
        }
```

### 2.8 sharding_method: ShardingMethod

**类型**: `ShardingMethod`

**说明**: 分片方法（自动分片或自定义分片）。

**枚举定义**:
```76:84:lib/collection/src/config.rs
#[derive(
    Debug, Deserialize, Serialize, JsonSchema, Anonymize, PartialEq, Eq, Hash, Clone, Copy, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum ShardingMethod {
    #[default]
    Auto,
    Custom,
}
```

**说明**:
- **Auto**: 自动分片，数据随机分布到所有分片
- **Custom**: 自定义分片，根据 `shard_key` 分布数据

---

## 三、核心方法

### 3.1 初始化方法

#### new()

```86:123:lib/collection/src/shards/shard_holder/mod.rs
    pub fn new(collection_path: &Path, sharding_method: ShardingMethod) -> CollectionResult<Self> {
        let shard_transfers =
            SaveOnDisk::load_or_init_default(collection_path.join(SHARD_TRANSFERS_FILE))?;
        let resharding_state: SaveOnDisk<Option<ReshardState>> =
            SaveOnDisk::load_or_init_default(collection_path.join(RESHARDING_STATE_FILE))?;

        let key_mapping: SaveOnDisk<ShardKeyMapping> =
            SaveOnDisk::load_or_init_default(collection_path.join(SHARD_KEY_MAPPING_FILE))?;

        // TODO(1.17.0): Remove once the old shardkey format has been removed entirely.
        Self::migrate_shard_key_if_needed(&key_mapping)?;

        let mut shard_id_to_key_mapping = AHashMap::new();

        for (shard_key, shard_ids) in key_mapping.read().iter() {
            for shard_id in shard_ids {
                shard_id_to_key_mapping.insert(*shard_id, shard_key.clone());
            }
        }

        let rings = match sharding_method {
            ShardingMethod::Auto => HashMap::from([(None, HashRingRouter::single())]),
            ShardingMethod::Custom => HashMap::new(),
        };

        let (shard_transfer_changes, _) = broadcast::channel(64);

        Ok(Self {
            shards: AHashMap::new(),
            shard_transfers,
            shard_transfer_changes,
            resharding_state,
            rings,
            key_mapping,
            shard_id_to_key_mapping,
            sharding_method,
        })
    }
```

**功能**: 创建新的 `ShardHolder` 实例。

**步骤**:
1. 加载或初始化分片转移任务
2. 加载或初始化重分片状态
3. 加载或初始化分片键映射
4. 迁移旧格式的分片键（如果需要）
5. 构建反向映射（`shard_id_to_key_mapping`）
6. 根据分片方法初始化哈希环
7. 创建分片转移变更广播通道

### 3.2 分片管理方法

#### add_shard()

```220:255:lib/collection/src/shards/shard_holder/mod.rs
    pub async fn add_shard(
        &mut self,
        shard_id: ShardId,
        shard: ShardReplicaSet,
        shard_key: Option<ShardKey>,
    ) -> CollectionResult<()> {
        let evicted = self.shards.insert(shard_id, shard);
        if let Some(evicted) = evicted {
            debug_assert!(false, "Overwriting existing shard id {shard_id}");
            evicted.stop_gracefully().await;
        }

        self.rings
            .entry(shard_key.clone())
            .or_insert_with(HashRingRouter::single)
            .add(shard_id);

        if let Some(shard_key) = shard_key {
            self.key_mapping.write_optional(|key_mapping| {
                let has_id = key_mapping
                    .get(&shard_key)
                    .map(|shard_ids| shard_ids.contains(&shard_id))
                    .unwrap_or(false);

                if has_id {
                    return None;
                }
                let mut copy_of_mapping = key_mapping.clone();
                let shard_ids = copy_of_mapping.entry(shard_key.clone()).or_default();
                shard_ids.insert(shard_id);
                Some(copy_of_mapping)
            })?;
            self.shard_id_to_key_mapping.insert(shard_id, shard_key);
        }
        Ok(())
    }
```

**功能**: 添加分片到 `ShardHolder`。

**步骤**:
1. 插入分片到 `shards` 映射（如果已存在则优雅停止旧分片）
2. 将分片 ID 添加到相应的哈希环
3. 如果有分片键，更新 `key_mapping` 和 `shard_id_to_key_mapping`

#### get_shard()

```357:359:lib/collection/src/shards/shard_holder/mod.rs
    pub fn get_shard(&self, shard_id: ShardId) -> Option<&ShardReplicaSet> {
        self.shards.get(&shard_id)
    }
```

**功能**: 根据分片 ID 获取分片副本集。

#### drop_and_remove_shard()

```177:196:lib/collection/src/shards/shard_holder/mod.rs
    pub async fn drop_and_remove_shard(&mut self, shard_id: ShardId) -> CollectionResult<()> {
        if let Some(replica_set) = self.shards.remove(&shard_id) {
            let shard_path = replica_set.shard_path.clone();
            replica_set.stop_gracefully().await;

            // Explicitly drop shard config file first
            // If removing all shard files at once, it may be possible for the shard configuration
            // file to be left behind if the process is killed in the middle. We must avoid this so
            // we don't attempt to load this shard anymore on restart.
            let shard_config_path = ShardConfig::get_config_path(&shard_path);
            if let Err(err) = tokio_fs::remove_file(shard_config_path).await {
                log::error!(
                    "Failed to remove shard config file before removing the rest of the files: {err}",
                );
            }

            tokio_fs::remove_dir_all(shard_path).await?;
        }
        Ok(())
    }
```

**功能**: 删除分片并移除其文件。

**步骤**:
1. 从 `shards` 中移除分片
2. 优雅停止分片副本集
3. 删除分片配置文件（优先删除，避免重启时加载）
4. 删除整个分片目录

### 3.3 分片路由方法

#### split_by_shard()

```377:426:lib/collection/src/shards/shard_holder/mod.rs
    pub fn split_by_shard<O: SplitByShard + Clone>(
        &self,
        operation: O,
        shard_keys_selection: &Option<ShardKey>,
    ) -> CollectionResult<Vec<(&ShardReplicaSet, O)>> {
        let Some(hashring) = self.rings.get(&shard_keys_selection.clone()) else {
            return if let Some(shard_key) = shard_keys_selection {
                Err(CollectionError::bad_input(format!(
                    "Shard key {shard_key} not found"
                )))
            } else {
                Err(CollectionError::bad_input(
                    "Shard key not specified".to_string(),
                ))
            };
        };

        if hashring.is_empty() {
            return Err(CollectionError::bad_input(
                "No shards found for shard key".to_string(),
            ));
        }

        let operation_to_shard = operation.split_by_shard(hashring);
        let shard_ops: Vec<_> = match operation_to_shard {
            OperationToShard::ByShard(by_shard) => by_shard
                .into_iter()
                .map(|(shard_id, operation)| (self.shards.get(&shard_id).unwrap(), operation))
                .collect(),
            OperationToShard::ToAll(operation) => {
                if let Some(shard_key) = shard_keys_selection {
                    let shard_ids = self
                        .key_mapping
                        .read()
                        .get(shard_key)
                        .cloned()
                        .unwrap_or_default();
                    shard_ids
                        .into_iter()
                        .map(|shard_id| (self.shards.get(&shard_id).unwrap(), operation.clone()))
                        .collect()
                } else {
                    self.all_shards()
                        .map(|shard| (shard, operation.clone()))
                        .collect()
                }
            }
        };
        Ok(shard_ops)
    }
```

**功能**: 将操作按分片拆分。

**步骤**:
1. 根据分片键选择相应的哈希环
2. 使用哈希环将操作拆分到不同的分片
3. 返回分片和操作的配对列表

### 3.4 分片转移方法

#### register_start_shard_transfer()

```428:436:lib/collection/src/shards/shard_holder/mod.rs
    pub fn register_start_shard_transfer(&self, transfer: ShardTransfer) -> CollectionResult<bool> {
        let changed = self
            .shard_transfers
            .write(|transfers| transfers.insert(transfer.clone()))?;
        let _ = self
            .shard_transfer_changes
            .send(ShardTransferChange::Start(transfer));
        Ok(changed)
    }
```

**功能**: 注册分片转移开始。

#### register_finish_transfer()

```438:446:lib/collection/src/shards/shard_holder/mod.rs
    pub fn register_finish_transfer(&self, key: &ShardTransferKey) -> CollectionResult<bool> {
        let any_removed = self
            .shard_transfers
            .write(|transfers| transfers.extract_if(|transfer| key.check(transfer)).count() > 0)?;
        let _ = self
            .shard_transfer_changes
            .send(ShardTransferChange::Finish(*key));
        Ok(any_removed)
    }
```

**功能**: 注册分片转移完成。

#### await_shard_transfer_end()

```462:497:lib/collection/src/shards/shard_holder/mod.rs
    pub fn await_shard_transfer_end(
        &self,
        transfer: ShardTransferKey,
        timeout: Duration,
    ) -> impl Future<Output = CollectionResult<Result<(), ()>>> {
        let mut subscriber = self.shard_transfer_changes.subscribe();
        let receiver = async move {
            loop {
                match subscriber.recv().await {
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return Err(CollectionError::service_error(
                            "Failed to await shard transfer end: failed to listen for shard transfer changes, channel closed",
                        ));
                    }
                    Err(err @ tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        return Err(CollectionError::service_error(format!(
                            "Failed to await shard transfer end: failed to listen for shard transfer changes, channel lagged behind: {err}",
                        )));
                    }
                    Ok(ShardTransferChange::Finish(key)) if key == transfer => return Ok(Ok(())),
                    Ok(ShardTransferChange::Abort(key)) if key == transfer => return Ok(Err(())),
                    Ok(_) => {}
                }
            }
        };

        async move {
            match tokio::time::timeout(timeout, receiver).await {
                Ok(operation) => Ok(operation?),
                // Timeout
                Err(err) => Err(CollectionError::service_error(format!(
                    "Awaiting for shard transfer end timed out: {err}"
                ))),
            }
        }
    }
```

**功能**: 等待分片转移结束（完成或中止）。

### 3.5 哈希环管理方法

#### rebuild_rings()

```282:323:lib/collection/src/shards/shard_holder/mod.rs
    fn rebuild_rings(&mut self) {
        let mut rings = match self.sharding_method {
            // With auto sharding, we have a single hash ring
            ShardingMethod::Auto => HashMap::from([(None, HashRingRouter::single())]),
            // With custom sharding, we have a hash ring per shard key
            ShardingMethod::Custom => HashMap::new(),
        };

        // Add shards and shard keys
        let ids_to_key = self.get_shard_id_to_key_mapping();
        for shard_id in self.shards.keys() {
            let shard_key = ids_to_key.get(shard_id).cloned();
            debug_assert!(
                matches!(
                    (self.sharding_method, &shard_key),
                    (ShardingMethod::Auto, None) | (ShardingMethod::Custom, Some(_)),
                ),
                "auto sharding cannot have shard key, custom sharding must have shard key ({:?}, {shard_key:?})",
                self.sharding_method,
            );
            rings
                .entry(shard_key)
                .or_insert_with(HashRingRouter::single)
                .add(*shard_id);
        }

        // Restore resharding hash ring if resharding is active and haven't reached
        // `WriteHashRingCommitted` stage yet
        if let Some(state) = self.resharding_state.read().deref() {
            let ring = rings
                .get_mut(&state.shard_key)
                .expect("must have hash ring for current resharding shard key");

            ring.start_resharding(state.shard_id, state.direction);

            if state.stage >= ReshardStage::WriteHashRingCommitted {
                ring.commit_resharding();
            }
        }

        self.rings = rings;
    }
```

**功能**: 重建哈希环。

**步骤**:
1. 根据分片方法初始化哈希环
2. 将所有分片添加到相应的哈希环
3. 如果正在进行重分片，恢复重分片状态

---

## 四、数据关系图

### 4.1 整体结构

```
ShardHolder
    │
    ├─ shards: AHashMap<ShardId, ShardReplicaSet>
    │   └─ 存储所有分片
    │
    ├─ shard_transfers: SaveOnDisk<HashSet<ShardTransfer>>
    │   └─ 持久化的分片转移任务
    │
    ├─ shard_transfer_changes: broadcast::Sender<ShardTransferChange>
    │   └─ 分片转移状态变更通知
    │
    ├─ resharding_state: SaveOnDisk<Option<ReshardState>>
    │   └─ 持久化的重分片状态
    │
    ├─ rings: HashMap<Option<ShardKey>, HashRingRouter>
    │   └─ 每个分片键的哈希环路由器
    │
    ├─ key_mapping: SaveOnDisk<ShardKeyMapping>
    │   └─ 分片键到分片 ID 的映射（持久化）
    │
    ├─ shard_id_to_key_mapping: AHashMap<ShardId, ShardKey>
    │   └─ 分片 ID 到分片键的反向映射（内存缓存）
    │
    └─ sharding_method: ShardingMethod
        └─ 分片方法（Auto 或 Custom）
```

### 4.2 分片键映射关系

```
ShardKeyMapping (key_mapping)
    └─ HashMap<ShardKey, HashSet<ShardId>>
        │
        ├─ "key1" → {0, 1, 2}
        ├─ "key2" → {3, 4}
        └─ ...

shard_id_to_key_mapping (反向映射)
    └─ AHashMap<ShardId, ShardKey>
        │
        ├─ 0 → "key1"
        ├─ 1 → "key1"
        ├─ 2 → "key1"
        ├─ 3 → "key2"
        └─ ...
```

### 4.3 哈希环路由关系

```
rings: HashMap<Option<ShardKey>, HashRingRouter>
    │
    ├─ Auto Sharding:
    │   └─ None → HashRingRouter::Single(HashRing)
    │       └─ 包含所有分片 ID
    │
    └─ Custom Sharding:
        ├─ Some("key1") → HashRingRouter::Single(HashRing)
        │   └─ 包含 key1 的分片 ID
        ├─ Some("key2") → HashRingRouter::Single(HashRing)
        │   └─ 包含 key2 的分片 ID
        └─ ...
```

---

## 五、持久化文件

### 5.1 文件列表

| 文件 | 路径 | 说明 |
|------|------|------|
| **shard_transfers** | `{collection_path}/shard_transfers` | 分片转移任务 |
| **resharding_state.json** | `{collection_path}/resharding_state.json` | 重分片状态 |
| **shard_key_mapping.json** | `{collection_path}/shard_key_mapping.json` | 分片键映射 |

### 5.2 文件格式

#### shard_key_mapping.json

```json
[
  {
    "key": "key1",
    "shard_ids": [0, 1, 2]
  },
  {
    "key": "key2",
    "shard_ids": [3, 4]
  }
]
```

---

## 六、使用场景

### 6.1 分片选择

当需要执行操作时，`ShardHolder` 使用哈希环路由选择目标分片：

1. 根据分片键（如果有）选择相应的哈希环
2. 使用哈希环的 `get()` 方法获取目标分片 ID
3. 从 `shards` 中获取对应的 `ShardReplicaSet`

### 6.2 分片转移

分片转移流程：

1. **开始转移**: `register_start_shard_transfer()` 注册转移任务
2. **监控转移**: `await_shard_transfer_end()` 等待转移完成
3. **完成转移**: `register_finish_transfer()` 标记转移完成
4. **中止转移**: `register_abort_transfer()` 中止转移

### 6.3 重分片

重分片流程：

1. 设置 `resharding_state` 为 `Some(ReshardState)`
2. 哈希环进入 `Resharding` 模式（包含 `old` 和 `new` 两个环）
3. 数据迁移完成后，提交重分片（`commit_resharding()`）
4. 清除 `resharding_state`

---

## 七、关键设计要点

### 7.1 双重映射设计

- **key_mapping**: 分片键 → 分片 ID 集合（持久化）
- **shard_id_to_key_mapping**: 分片 ID → 分片键（内存缓存）

**优势**:
- 持久化保证数据不丢失
- 内存缓存提供快速查找
- 两个映射保持同步

### 7.2 哈希环设计

- **Auto Sharding**: 单个哈希环，所有分片共享
- **Custom Sharding**: 每个分片键独立哈希环

**优势**:
- 支持不同分片键的独立路由
- 支持重分片时的双环模式

### 7.3 持久化设计

- 使用 `SaveOnDisk` 包装，自动持久化
- 关键状态（转移、重分片、映射）都持久化
- 支持从磁盘恢复状态

---

## 八、代码位置总结

### 8.1 核心文件

| 组件 | 文件 | 说明 |
|------|------|------|
| **ShardHolder 定义** | `lib/collection/src/shards/shard_holder/mod.rs` | 主要实现 |
| **ShardKeyMapping** | `lib/collection/src/shards/shard_holder/shard_mapping.rs` | 分片键映射 |
| **HashRingRouter** | `lib/collection/src/hash_ring.rs` | 哈希环路由器 |
| **ShardReplicaSet** | `lib/collection/src/shards/replica_set/mod.rs` | 分片副本集 |

### 8.2 相关类型

| 类型 | 定义位置 | 说明 |
|------|---------|------|
| **ShardId** | `lib/collection/src/shards/shard.rs` | `u32` |
| **ShardKey** | `lib/segment/src/types.rs` | 分片键（Keyword 或 Number） |
| **ShardingMethod** | `lib/collection/src/config.rs` | Auto 或 Custom |
| **ShardTransfer** | `lib/collection/src/shards/transfer/mod.rs` | 分片转移任务 |

---

## 九、总结

### 9.1 核心职责

1. **分片存储**: 管理所有分片的映射
2. **分片路由**: 使用哈希环将操作路由到正确的分片
3. **分片转移**: 管理分片在节点间的转移
4. **重分片**: 支持动态增加或减少分片数量
5. **状态持久化**: 持久化关键状态，支持恢复

### 9.2 关键特点

1. **高性能**: 使用 `AHashMap` 和内存缓存提供快速访问
2. **持久化**: 关键状态自动持久化到磁盘
3. **并发安全**: 使用 `RwLock` 保护并发访问
4. **事件通知**: 使用广播通道通知状态变更
5. **灵活路由**: 支持自动和自定义两种分片模式

---

## 参考资料

- [Collection、Shard、Segment 关系详解](./Collection_Shard_Segment关系详解.md)
- [Qdrant 协调节点实现解读](./Qdrant协调节点实现解读.md)
- [Collection 元数据存储位置说明](./Collection元数据存储位置说明.md)
