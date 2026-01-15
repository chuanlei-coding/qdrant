# Qdrant 的 Raft 实现说明

## 概述

**Qdrant 使用了第三方 Raft 库**，具体是 **raft-rs**（由 TiKV 项目开发）。

## 证据

### 1. Cargo.toml 依赖声明

在 `Cargo.toml` 中明确声明了 Raft 依赖：

```toml
# Consensus related crates
raft = { workspace = true }
raft-proto = { version = "0.7.0", features = [
    "prost-codec",
], default-features = false }

# 在 workspace.dependencies 中
raft = { version = "0.7.0", features = ["prost-codec"], default-features = false }
prost-for-raft = { package = "prost", version = "=0.11.9" } # version of prost used by raft
```

### 2. 代码中的使用

在 `src/consensus.rs` 中可以看到：

```rust
use raft::eraftpb::Message as RaftMessage;
use raft::prelude::*;
use raft::{INVALID_ID, SoftState, StateRole};

type Node = RawNode<ConsensusStateRef>;
```

这些都是 `raft-rs` 库的典型 API。

### 3. Storage Trait 实现

Qdrant 实现了 `raft::Storage` trait，这是 raft-rs 库要求的接口：

```rust
// lib/storage/src/content_manager/consensus_manager.rs

impl Storage for ConsensusStateRef {
    fn initial_state(&self) -> raft::Result<RaftState> {
        self.0.initial_state()
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        context: GetEntriesContext,
    ) -> raft::Result<Vec<RaftEntry>> {
        self.0.entries(low, high, max_size, context)
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        self.0.term(idx)
    }

    fn first_index(&self) -> raft::Result<EntryId> {
        self.0.first_index()
    }

    fn last_index(&self) -> raft::Result<EntryId> {
        self.0.last_index()
    }

    fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<raft::eraftpb::Snapshot> {
        self.0.snapshot(request_index, to)
    }
}
```

### 4. RawNode 的使用

Qdrant 使用 `RawNode` 来管理 Raft 节点：

```rust
// src/consensus.rs
type Node = RawNode<ConsensusStateRef>;

pub struct Consensus {
    /// Raft structure which handles raft-related state
    node: Node,
    // ...
}

// 创建 Raft 节点
let mut node = Node::new(&raft_config, state_ref.clone(), logger)?;
node.set_batch_append(true);
```

---

## raft-rs 库简介

### 库信息

- **名称**：raft-rs
- **版本**：0.7.0（Qdrant 使用的版本）
- **来源**：TiKV 项目
- **GitHub**：https://github.com/tikv/raft-rs
- **许可证**：Apache-2.0

### 库特点

1. **生产就绪**：TiKV 在生产环境中广泛使用
2. **性能优化**：经过大量优化和测试
3. **功能完整**：实现了完整的 Raft 算法
4. **可扩展**：提供了灵活的接口

### 核心组件

- **RawNode**：Raft 节点的核心结构
- **Storage**：存储接口 trait
- **RaftState**：Raft 状态
- **Message**：Raft 消息类型

---

## Qdrant 的 Raft 集成架构

### 架构层次

```
应用层（Qdrant）
  ↓
Consensus 模块（src/consensus.rs）
  ↓
raft-rs 库（RawNode）
  ↓
Storage 实现（ConsensusStateRef）
  ↓
持久化层（WAL + Snapshot）
```

### 关键组件

#### 1. Consensus 结构

```rust
pub struct Consensus {
    /// Raft structure which handles raft-related state
    node: Node,  // RawNode<ConsensusStateRef>
    /// Receives proposals from peers and client for applying in consensus
    receiver: Receiver<Message>,
    /// Runtime for async message sending
    runtime: Handle,
    config: ConsensusConfig,
    broker: RaftMessageBroker,
    raft_config: Config,
}
```

#### 2. Storage 实现

Qdrant 通过 `ConsensusManager` 和 `ConsensusStateRef` 实现了 `Storage` trait：

```rust
pub struct ConsensusManager<C: CollectionContainer> {
    pub persistent: RwLock<Persistent>,  // 持久化状态
    pub is_leader_established: Arc<IsReady>,
    wal: Mutex<ConsensusOpWal>,  // WAL（预写日志）
    soft_state: RwLock<Option<SoftState>>,  // 软状态
    toc: Arc<C>,  // 集合容器
    // ...
}
```

#### 3. 消息处理

```rust
pub enum Message {
    FromClient(ConsensusOperations),  // 来自客户端的操作
    FromPeer(Box<RaftMessage>),       // 来自其他节点的 Raft 消息
}
```

---

## Qdrant 的 Raft 实现细节

### 1. 初始化

```rust
// src/consensus.rs:246
let mut node = Node::new(&raft_config, state_ref.clone(), logger)?;
node.set_batch_append(true);
```

### 2. 消息处理循环

```rust
// src/consensus.rs
loop {
    // 获取 Raft Ready
    let ready = self.node.ready();
    
    // 处理 Ready 中的消息
    // - 发送消息到其他节点
    // - 应用已提交的条目
    // - 处理快照
    // - 更新状态
    
    // 推进 Raft 状态
    self.node.advance(ready);
}
```

### 3. 存储接口实现

Qdrant 实现了以下 Storage 方法：

- `initial_state()`：获取初始 Raft 状态
- `entries()`：获取指定范围的日志条目
- `term()`：获取指定索引的任期
- `first_index()`：获取第一个日志索引
- `last_index()`：获取最后一个日志索引
- `snapshot()`：创建快照

### 4. WAL（预写日志）

Qdrant 使用自定义的 WAL 实现来持久化 Raft 日志：

```rust
// lib/storage/src/content_manager/consensus/consensus_wal.rs
pub struct ConsensusOpWal {
    // WAL 实现
}
```

### 5. 快照

Qdrant 实现了快照功能，用于压缩日志：

```rust
pub struct SnapshotData {
    pub collections_data: CollectionsSnapshot,
    pub address_by_id: PeerAddressById,
    pub metadata_by_id: PeerMetadataById,
    pub cluster_metadata: HashMap<String, serde_json::Value>,
}
```

---

## 为什么使用第三方库？

### 优势

1. **成熟稳定**：raft-rs 是经过生产验证的库
2. **减少开发成本**：无需从零实现 Raft 算法
3. **社区支持**：有活跃的社区和维护
4. **性能优化**：库已经过大量优化
5. **功能完整**：实现了完整的 Raft 协议

### Qdrant 的定制化

虽然使用了第三方库，但 Qdrant 在以下方面进行了定制：

1. **Storage 实现**：自定义的存储层（WAL + Snapshot）
2. **消息传输**：使用 gRPC 进行节点间通信
3. **操作封装**：将业务操作封装为 `ConsensusOperations`
4. **状态管理**：自定义的状态管理和持久化

---

## 总结

### 关键点

1. ✅ **Qdrant 使用了第三方 Raft 库**：raft-rs（版本 0.7.0）
2. ✅ **库来源**：TiKV 项目开发的 raft-rs
3. ✅ **集成方式**：通过实现 `Storage` trait 集成
4. ✅ **定制化**：Qdrant 实现了自己的存储层和消息传输

### 架构特点

- **分层设计**：应用层 → Consensus 层 → raft-rs 库 → Storage 层
- **接口抽象**：通过 Storage trait 实现存储抽象
- **灵活扩展**：可以自定义存储和传输机制

### 代码位置

- **Consensus 实现**：`src/consensus.rs`
- **Storage 实现**：`lib/storage/src/content_manager/consensus_manager.rs`
- **WAL 实现**：`lib/storage/src/content_manager/consensus/consensus_wal.rs`
- **持久化状态**：`lib/storage/src/content_manager/consensus/persistent.rs`

---

## 参考资料

- [raft-rs GitHub](https://github.com/tikv/raft-rs)
- [raft-rs 文档](https://docs.rs/raft/)
- [Qdrant 分布式部署文档](https://qdrant.tech/documentation/guides/distributed_deployment/)
- [Raft 算法论文](https://raft.github.io/raft.pdf)

---

**最后更新**：2025年1月
