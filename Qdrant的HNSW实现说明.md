# Qdrant 的 HNSW 实现说明

## 概述

**Qdrant 的 HNSW（Hierarchical Navigable Small World）算法是自己实现的**，而不是使用第三方库。

## 证据

### 1. 代码结构

Qdrant 有完整的 HNSW 实现目录结构：

```
lib/segment/src/index/hnsw_index/
├── hnsw.rs                    # HNSW 索引主实现
├── graph_layers.rs            # 图层次结构实现
├── graph_layers_builder.rs    # 图构建器实现
├── graph_links.rs             # 图链接实现
├── graph_layers_healer.rs     # 图修复器
├── point_scorer.rs            # 点评分器
├── entry_points.rs            # 入口点管理
├── search_context.rs          # 搜索上下文
├── build_cache.rs             # 构建缓存
├── config.rs                  # 配置管理
├── links_container.rs         # 链接容器
├── gpu/                       # GPU 加速支持（可选）
│   ├── gpu_graph_builder.rs
│   ├── gpu_vector_storage/
│   └── ...
└── tests/                     # 测试文件
```

### 2. 依赖检查

在 `Cargo.toml` 中**没有找到任何 HNSW 相关的第三方依赖**：

- `lib/segment/Cargo.toml` - 没有 HNSW 库依赖
- `Cargo.toml`（根目录）- 没有 HNSW 库依赖

### 3. 代码实现细节

#### 3.1 核心实现文件

**`lib/segment/src/index/hnsw_index/hnsw.rs`**：

```rust
#[derive(Debug)]
pub struct HNSWIndex {
    id_tracker: Arc<AtomicRefCell<IdTrackerSS>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    config: HnswGraphConfig,
    path: PathBuf,
    graph: GraphLayers,  // 自己实现的图结构
    searches_telemetry: HNSWSearchesTelemetry,
    is_on_disk: bool,
}
```

**`lib/segment/src/index/hnsw_index/graph_layers.rs`**：

```rust
//! # Search on level functions
//!
//! This module contains multiple variations of the SEARCH-LAYER function.
//! All of them implement a beam (greedy) search for closest points within a
//! single graph layer.
//!
//! - [`GraphLayersBase::search_on_level`]
//!   Regular search, as described in the original HNSW paper.
//!   Usually used on layer 0.
```

注释明确提到这是基于**原始 HNSW 论文**的实现。

#### 3.2 图构建器

**`lib/segment/src/index/hnsw_index/graph_layers_builder.rs`**：

```rust
/// Same as `GraphLayers`,  but allows to build in parallel
/// Convertible to `GraphLayers`
pub struct GraphLayersBuilder {
    max_level: AtomicUsize,
    hnsw_m: HnswM,
    ef_construct: usize,
    level_factor: f64,
    use_heuristic: bool,
    links_layers: Vec<LockedLayersContainer>,
    entry_points: Mutex<EntryPoints>,
    visited_pool: VisitedPool,
    ready_list: BitVec<AtomicUsize>,
}
```

这是 Qdrant 自己实现的并行图构建器。

### 4. 功能特性

Qdrant 的 HNSW 实现包含以下特性：

#### 4.1 核心功能

- ✅ **分层图结构**：实现了完整的 HNSW 分层导航小世界图
- ✅ **并行构建**：支持多线程并行构建索引
- ✅ **增量构建**：支持增量式构建（incremental building）
- ✅ **图修复**：实现了 `GraphLayersHealer` 用于修复图结构
- ✅ **过滤搜索**：支持带过滤条件的搜索
- ✅ **量化支持**：支持量化向量的搜索

#### 4.2 高级特性

- ✅ **GPU 加速**：可选 GPU 加速支持（通过 `gpu` feature）
- ✅ **内存映射**：支持 mmap 存储图结构
- ✅ **链接压缩**：支持压缩图链接以节省空间
- ✅ **内联向量存储**：支持将向量内联存储在图中
- ✅ **ACORN 算法**：实现了 ACORN-1 搜索算法变体

#### 4.3 优化特性

- ✅ **启发式优化**：使用启发式方法优化图构建
- ✅ **访问列表池**：使用 `VisitedPool` 复用访问列表
- ✅ **批量搜索**：支持批量向量搜索
- ✅ **多向量支持**：支持多向量（multivector）索引

### 5. 测试和基准测试

Qdrant 为 HNSW 实现提供了完整的测试和基准测试：

- **单元测试**：`lib/segment/src/index/hnsw_index/tests/`
- **基准测试**：
  - `hnsw_build_graph` - 图构建性能测试
  - `hnsw_search_graph` - 图搜索性能测试
  - `hnsw_build_asymptotic` - 渐近性能测试
  - `hnsw_incremental_build` - 增量构建测试

### 6. 代码规模

根据代码结构，Qdrant 的 HNSW 实现包含：

- **主要实现文件**：10+ 个 Rust 源文件
- **GPU 支持**：10+ 个 GPU 相关文件（可选）
- **测试文件**：多个测试和基准测试文件
- **代码行数**：估计数千行 Rust 代码

---

## 为什么自己实现？

### 优势

1. **完全控制**：可以针对 Qdrant 的特定需求进行优化
2. **性能优化**：可以深度优化以匹配 Qdrant 的存储和查询模式
3. **功能定制**：可以添加 Qdrant 特有的功能（如过滤搜索、量化支持等）
4. **集成深度**：与 Qdrant 的其他组件（存储、索引、查询）深度集成
5. **无依赖**：减少外部依赖，提高可控性

### 定制化特性

Qdrant 的 HNSW 实现包含许多定制化特性：

1. **过滤搜索**：在 HNSW 搜索过程中集成 payload 过滤
2. **量化集成**：深度集成向量量化（PQ、SQ、BQ）
3. **多向量支持**：支持每个点有多个向量
4. **GPU 加速**：可选的 GPU 加速构建和搜索
5. **增量构建**：支持增量式构建，无需重建整个索引
6. **图修复**：自动修复图结构中的问题

---

## 实现参考

### 算法基础

Qdrant 的 HNSW 实现基于：

- **原始 HNSW 论文**：Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
- **ACORN 算法**：实现了 ACORN-1 变体用于搜索优化

### 关键组件

1. **GraphLayers**：图层次结构，管理不同层的图
2. **GraphLayersBuilder**：并行图构建器
3. **GraphLinks**：图链接存储（支持多种格式）
4. **EntryPoints**：入口点管理
5. **VisitedPool**：访问列表池，用于搜索优化
6. **FilteredScorer**：带过滤的评分器

---

## 总结

### 关键点

1. ✅ **Qdrant 的 HNSW 是自己实现的**，不是使用第三方库
2. ✅ **完整实现**：包含完整的 HNSW 算法实现
3. ✅ **深度定制**：针对 Qdrant 的需求进行了大量定制和优化
4. ✅ **功能丰富**：包含过滤搜索、量化支持、GPU 加速等高级特性
5. ✅ **生产就绪**：经过充分测试和优化，用于生产环境

### 代码位置

- **主要实现**：`lib/segment/src/index/hnsw_index/`
- **核心文件**：
  - `hnsw.rs` - HNSW 索引主实现
  - `graph_layers.rs` - 图层次结构
  - `graph_layers_builder.rs` - 图构建器
  - `graph_links.rs` - 图链接存储

### 与其他组件的集成

- **向量存储**：与 `VectorStorage` 集成
- **量化**：与 `QuantizedVectors` 集成
- **Payload 索引**：与 `StructPayloadIndex` 集成
- **ID 跟踪**：与 `IdTrackerSS` 集成

---

## 参考资料

- [HNSW 论文](https://arxiv.org/abs/1603.09320)
- [Qdrant HNSW 文档](https://qdrant.tech/documentation/concepts/indexing/)
- [Qdrant 源码 - HNSW 实现](https://github.com/qdrant/qdrant/tree/master/lib/segment/src/index/hnsw_index)

---

**最后更新**：2025年1月
