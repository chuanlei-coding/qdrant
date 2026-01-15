# Qdrant 中索引的存储方式说明

## 概述

Qdrant 中的索引主要分为两大类：
1. **向量索引（Vector Index）**：主要是 HNSW 索引，用于加速向量相似度搜索
2. **Payload 索引（Payload Index）**：用于加速基于 payload 字段的过滤查询，包括数值索引、Map 索引、Geo 索引、全文索引等

所有索引都支持两种存储模式：
- **内存存储（RAM）**：索引数据完全加载到内存中，访问速度快，但占用内存
- **磁盘存储（Disk/Mmap）**：索引数据存储在磁盘上，通过内存映射（mmap）访问，节省内存，但访问速度相对较慢

## 历史版本说明

### 早期版本（使用 RocksDB）

在早期版本中，当使用 RocksDB 作为向量和 payload 的存储引擎时：

1. **HNSW 索引**：**始终存储在独立的文件中**，不存储在 RocksDB 中
   - 存储位置：segment 目录下的 `index/` 子目录
   - 存储文件：`graph.bin` 和 `links.bin`（或压缩版本）
   - **与是否使用 RocksDB 无关**

2. **Payload 索引**：**存储在 RocksDB 的列族（Column Family）中**
   - 每个字段的索引对应一个列族
   - 列族命名规则：
     - 数值索引：`{field}_numeric`
     - Map 索引：`{field}_map`
     - Geo 索引：`{field}_geo`
     - 全文索引：`{field}_text`
   - 索引数据在内存中构建，然后序列化存储到 RocksDB
   - 加载时，从 RocksDB 读取数据并重建内存索引

**注意**：Qdrant 正在逐步迁移远离 RocksDB，新版本默认使用 mmap 存储。

---

## 1. HNSW 索引存储

### 1.1 存储配置

HNSW 索引的存储方式由 `HnswConfig` 中的 `on_disk` 参数控制：

```rust
pub struct HnswConfig {
    pub m: usize,
    pub ef_construct: usize,
    pub full_scan_threshold: usize,
    pub max_indexing_threads: usize,
    /// Store HNSW index on disk. If set to false, index will be stored in RAM. Default: false
    pub on_disk: Option<bool>,
    // ...
}
```

**位置**: `lib/segment/src/types.rs`

### 1.2 存储文件

HNSW 索引存储在 segment 目录下的 `index/` 子目录中，包含以下文件：

#### 1.2.1 `graph.bin` - 图元数据

存储图的配置信息和入口点：

```rust
#[derive(Deserialize, Serialize, Debug)]
struct GraphLayerData {
    pub m: usize,              // 每层的最大连接数
    pub m0: usize,             // 第 0 层的最大连接数
    pub ef_construct: usize,    // 构建时的搜索范围
    pub entry_points: EntryPoints, // 图的入口点
}
```

**位置**: `lib/segment/src/index/hnsw_index/graph_layers.rs`

#### 1.2.2 链接文件 - 图的连接关系

根据压缩格式，可能使用以下文件之一：

- **`links.bin`** - 未压缩格式（Plain）
- **`links_compressed.bin`** - 压缩格式（Compressed）
- **`links_comp_vec.bin`** - 带向量的压缩格式（CompressedWithVectors）

**位置**: `lib/segment/src/index/hnsw_index/graph_links.rs`

### 1.3 存储实现

#### 1.3.1 磁盘存储（`on_disk=true`）

当 `on_disk=true` 时，HNSW 索引使用 **内存映射文件（mmap）** 存储：

```rust
pub fn load_from_file(
    path: &Path,
    on_disk: bool,
    format: GraphLinksFormat,
) -> OperationResult<Self> {
    let populate = !on_disk;
    let mmap = open_read_mmap(path, AdviceSetting::Advice(Advice::Random), populate)?;
    Self::try_new(GraphLinksEnum::Mmap(Arc::new(mmap)), |x| {
        GraphLinksView::load(x.as_bytes(), format)
    })
}
```

**特点**：
- 使用 `memmap2::Mmap` 进行内存映射
- 使用 `Advice::Random` 提示操作系统随机访问模式
- 如果 `populate=false`，数据按需从磁盘加载到内存
- 节省内存，适合大规模索引

**位置**: `lib/segment/src/index/hnsw_index/graph_links.rs:225-235`

#### 1.3.2 内存存储（`on_disk=false`）

当 `on_disk=false` 时，HNSW 索引完全存储在内存中：

```rust
pub fn new_from_edges(
    edges: Vec<Vec<Vec<PointOffsetType>>>,
    format_param: GraphLinksFormatParam<'_>,
    hnsw_m: HnswM,
) -> OperationResult<Self> {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    serialize_graph_links(edges, format_param, hnsw_m, &mut cursor)?;
    let mut bytes = cursor.into_inner();
    bytes.shrink_to_fit();
    Self::try_new(GraphLinksEnum::Ram(bytes), |x| {
        GraphLinksView::load(x.as_bytes(), format_param.as_format())
    })
}
```

**特点**：
- 数据存储在 `Vec<u8>` 中
- 所有数据都在内存中，访问速度快
- 占用内存较大，适合小到中等规模的索引

**位置**: `lib/segment/src/index/hnsw_index/graph_links.rs:237-249`

#### 1.3.3 构建时的存储选择

在构建 HNSW 索引时，根据 `on_disk` 参数选择存储方式：

```rust
pub fn into_graph_layers(
    self,
    path: &Path,
    format_param: GraphLinksFormatParam,
    on_disk: bool,
) -> OperationResult<GraphLayers> {
    let links_path = GraphLayers::get_links_path(path, format_param.as_format());
    let edges = Self::links_layers_to_edges(self.links_layers);
    let links;
    if on_disk {
        // Save memory by serializing directly to disk, then re-loading as mmap.
        atomic_save(&links_path, |writer| {
            serialize_graph_links(edges, format_param, self.hnsw_m, writer)
        })?;
        links = GraphLinks::load_from_file(&links_path, true, format_param.as_format())?;
    } else {
        // Since we'll keep it in the RAM anyway, we can afford to build in the RAM too.
        links = GraphLinks::new_from_edges(edges, format_param, self.hnsw_m)?;
        atomic_save(&links_path, |writer| writer.write_all(links.as_bytes()))?;
    }
    // ...
}
```

**位置**: `lib/segment/src/index/hnsw_index/graph_layers_builder.rs:205-243`

---

## 2. Payload 索引存储

Payload 索引用于加速基于 payload 字段的过滤查询。所有 payload 索引都使用 **mmap 存储**，但可以选择是否预加载到内存。

### 2.1 数值索引（Numeric Index）

#### 2.1.1 存储结构

```rust
pub struct MmapNumericIndex<T: Encodable + Numericable + Default + MmapValue + 'static> {
    path: PathBuf,
    pub(super) storage: Storage<T>,
    histogram: Histogram<T>,
    deleted_count: usize,
    max_values_per_point: usize,
    is_on_disk: bool,
}

pub(super) struct Storage<T: Encodable + Numericable + Default + MmapValue + 'static> {
    deleted: MmapBitSliceBufferedUpdateWrapper,  // 删除标记
    pairs: MmapSlice<Point<T>>,                  // 排序的 (id, value) 对
    pub(super) point_to_values: MmapPointToValues<T>, // 点到值的映射
}
```

**位置**: `lib/segment/src/index/field_index/numeric_index/mmap_numeric_index.rs`

#### 2.1.2 存储文件

- **`data.bin`** - 排序的 (point_id, value) 对
- **`deleted.bin`** - 删除标记的位图
- **`point_to_values.bin`** - 点到值的映射关系
- **`mmap_field_index_config.json`** - 索引配置

### 2.2 Map 索引（Map Index）

#### 2.2.1 存储结构

```rust
pub struct MmapMapIndex<N: MapIndexKey + Key + ?Sized> {
    path: PathBuf,
    pub(super) storage: Storage<N>,
    deleted_count: usize,
    total_key_value_pairs: usize,
    is_on_disk: bool,
}

pub(super) struct Storage<N: MapIndexKey + Key + ?Sized> {
    pub(super) value_to_points: MmapHashMap<N, PointOffsetType>, // 值到点的映射
    point_to_values: MmapPointToValues<N>,                       // 点到值的映射
    pub(super) deleted: MmapBitSliceBufferedUpdateWrapper,        // 删除标记
}
```

**位置**: `lib/segment/src/index/field_index/map_index/mmap_map_index.rs`

#### 2.2.2 存储文件

- **`values_to_points.bin`** - 值到点的哈希映射
- **`point_to_values.bin`** - 点到值的映射关系
- **`deleted.bin`** - 删除标记的位图
- **`mmap_field_index_config.json`** - 索引配置

### 2.3 Geo 索引（Geo Index）

#### 2.3.1 存储结构

```rust
pub struct MmapGeoIndex {
    path: PathBuf,
    pub(super) storage: Storage,
    deleted_count: usize,
    is_on_disk: bool,
}

pub(super) struct Storage {
    deleted: MmapBitSliceBufferedUpdateWrapper,
    counts_per_hash: MmapSlice<Counts>,        // 每个 GeoHash 的点数和值数
    points_map: MmapSlice<PointKeyValue>,      // 点到 GeoHash 的映射
    points_map_ids: MmapSlice<PointOffsetType>, // 点的 ID 列表
}
```

**位置**: `lib/segment/src/index/field_index/geo_index/mmap_geo_index.rs`

#### 2.3.2 存储文件

- **`counts_per_hash.bin`** - 每个 GeoHash 的统计信息
- **`points_map.bin`** - 点到 GeoHash 的映射
- **`points_map_ids.bin`** - 点的 ID 列表
- **`deleted.bin`** - 删除标记的位图
- **`mmap_field_index_stats.json`** - 索引统计信息

### 2.4 全文索引（Full-Text Index）

#### 2.4.1 存储结构

```rust
pub struct MmapInvertedIndex {
    path: PathBuf,
    pub(super) storage: Storage,
    is_on_disk: bool,
}

pub(in crate::index::field_index::full_text_index) struct Storage {
    pub(in crate::index::field_index::full_text_index) postings: MmapPostingsEnum,      // 倒排列表
    pub(in crate::index::field_index::full_text_index) vocab: MmapHashMap<str, TokenId>, // 词汇表
    pub(in crate::index::field_index::full_text_index) point_to_tokens_count: MmapSlice<usize>, // 点到 token 数的映射
    pub(in crate::index::field_index::full_text_index) deleted_points: MmapBitSliceBufferedUpdateWrapper, // 删除标记
}
```

**位置**: `lib/segment/src/index/field_index/full_text_index/inverted_index/mmap_inverted_index/mod.rs`

#### 2.4.2 存储文件

- **`postings.bin`** - 倒排列表（postings list）
- **`vocab.bin`** - 词汇表（token -> token_id 的映射）
- **`point_to_tokens_count.bin`** - 每个点的 token 数量
- **`deleted_points.bin`** - 删除标记的位图

### 2.5 通用存储组件

#### 2.5.1 `MmapPointToValues` - 点到值的映射

所有 payload 索引都使用 `MmapPointToValues` 来存储点到值的映射关系：

```rust
pub struct MmapPointToValues<T: MmapValue + ?Sized> {
    file_name: PathBuf,
    mmap: Mmap,
    header: Header,
    phantom: std::marker::PhantomData<T>,
}
```

**特点**：
- 使用 mmap 存储扁平化的点到值映射
- 类似于 `Vec<Vec<T>>`，但存储在内存映射文件中
- 支持按需加载，节省内存

**位置**: `lib/segment/src/index/field_index/mmap_point_to_values.rs`

#### 2.5.2 `MmapBitSlice` - 删除标记

使用位图来标记已删除的点：

```rust
pub struct MmapBitSlice {
    mmap: Mmap,
    len: usize,
}
```

**位置**: `lib/common/memory/src/mmap_type.rs`

---

## 3. 存储方式对比

### 3.1 内存存储 vs 磁盘存储

| 特性 | 内存存储（RAM） | 磁盘存储（Mmap） |
|------|----------------|------------------|
| **访问速度** | 极快 | 较快（按需加载） |
| **内存占用** | 高（全部加载） | 低（按需加载） |
| **适用场景** | 小到中等规模索引 | 大规模索引 |
| **持久化** | 需要定期保存到磁盘 | 直接持久化 |
| **启动速度** | 需要从磁盘加载 | 直接使用磁盘文件 |

### 3.2 HNSW 索引的存储选择建议

- **`on_disk=false`（内存存储）**：
  - 索引大小 < 几 GB
  - 需要极快的搜索速度
  - 有充足的内存资源

- **`on_disk=true`（磁盘存储）**：
  - 索引大小 > 几 GB
  - 内存资源有限
  - 可以接受稍慢的搜索速度（但通过 mmap 仍然很快）

### 3.3 Payload 索引的存储

Payload 索引默认使用 mmap 存储，但可以通过 `populate` 参数控制是否预加载到内存：

- **`populate=false`**：按需加载，节省内存
- **`populate=true`**：预加载到内存，提高访问速度

---

## 4. 存储优化技术

### 4.1 内存映射（mmap）

Qdrant 使用 `memmap2` 库进行内存映射，提供以下优势：

- **零拷贝访问**：直接访问磁盘文件，无需额外的内存拷贝
- **按需加载**：操作系统按需将页面加载到内存
- **共享内存**：多个进程可以共享同一个 mmap 文件

### 4.2 内存访问建议（madvise）

Qdrant 使用 `madvise` 系统调用来优化内存访问模式：

- **`Advice::Random`**：提示操作系统随机访问模式（用于 HNSW 索引）
- **`Advice::Sequential`**：提示操作系统顺序访问模式（用于顺序扫描）
- **`Advice::Normal`**：默认访问模式

**位置**: `lib/common/memory/src/madvise.rs`

### 4.3 压缩格式

HNSW 索引支持三种压缩格式：

1. **Plain（未压缩）**：原始格式，访问最快，但占用空间最大
2. **Compressed（压缩）**：压缩链接数据，节省空间，访问稍慢
3. **CompressedWithVectors（带向量压缩）**：同时压缩链接和向量数据，最节省空间

---

## 5. 代码位置总结

### 5.1 HNSW 索引

- **配置**: `lib/segment/src/types.rs` - `HnswConfig`
- **图元数据**: `lib/segment/src/index/hnsw_index/graph_layers.rs`
- **链接存储**: `lib/segment/src/index/hnsw_index/graph_links.rs`
- **构建器**: `lib/segment/src/index/hnsw_index/graph_layers_builder.rs`

### 5.2 Payload 索引

- **数值索引**: `lib/segment/src/index/field_index/numeric_index/mmap_numeric_index.rs`
- **Map 索引**: `lib/segment/src/index/field_index/map_index/mmap_map_index.rs`
- **Geo 索引**: `lib/segment/src/index/field_index/geo_index/mmap_geo_index.rs`
- **全文索引**: `lib/segment/src/index/field_index/full_text_index/inverted_index/mmap_inverted_index/mod.rs`
- **通用组件**: `lib/segment/src/index/field_index/mmap_point_to_values.rs`

### 5.3 底层存储

- **mmap 操作**: `lib/common/memory/src/mmap_ops.rs`
- **mmap 类型**: `lib/common/memory/src/mmap_type.rs`
- **内存建议**: `lib/common/memory/src/madvise.rs`

---

## 6. 早期版本中的 RocksDB 索引存储

### 6.1 HNSW 索引（与 RocksDB 无关）

**重要**：HNSW 索引的存储方式与是否使用 RocksDB **完全无关**。无论向量和 payload 数据存储在 RocksDB 还是 mmap 中，HNSW 索引都存储在独立的文件中。

**存储位置**：
- segment 目录下的 `index/` 子目录
- 文件：`graph.bin` 和 `links.bin`（或压缩版本）

**原因**：
- HNSW 图结构不适合存储在键值数据库中
- 需要高效的随机访问和遍历
- 文件存储提供更好的性能

### 6.2 Payload 索引（存储在 RocksDB 列族中）

在早期版本中，当使用 RocksDB 时，payload 索引存储在 RocksDB 的列族中。

#### 6.2.1 存储结构

```rust
// 数值索引存储在列族中
fn numeric_index_storage_cf_name(field: &str) -> String {
    format!("{field}_numeric")
}

// Map 索引存储在列族中
pub fn storage_cf_name(field: &str) -> String {
    format!("{field}_map")
}

// Geo 索引存储在列族中
fn storage_cf_name(field: &str) -> String {
    format!("{field}_geo")
}
```

**位置**: 
- `lib/segment/src/index/field_index/numeric_index/mod.rs:1248`
- `lib/segment/src/index/field_index/map_index/mod.rs:278`
- `lib/segment/src/index/field_index/geo_index/mod.rs:172`

#### 6.2.2 存储方式

**数值索引**：
```rust
/// Open and load mutable numeric index from RocksDB storage
pub fn open_rocksdb(
    db: Arc<RwLock<DB>>,
    field: &str,
    create_if_missing: bool,
) -> OperationResult<Option<Self>> {
    let store_cf_name = super::numeric_index_storage_cf_name(field);
    let db_wrapper = DatabaseColumnScheduledDeleteWrapper::new(
        DatabaseColumnWrapper::new(db, &store_cf_name)
    );
    // Load in-memory index from RocksDB
    let in_memory_index = db_wrapper
        .lock_db()
        .iter()?
        .map(|(key, value)| {
            // 从 RocksDB 键值对重建内存索引
            let value_idx = u32::from_be_bytes(value.as_ref().try_into()?);
            let (idx, value) = T::decode_key(&key);
            Ok((idx, value))
        })
        .collect::<Result<InMemoryNumericIndex<_>, OperationError>>()?;
    // ...
}
```

**位置**: `lib/segment/src/index/field_index/numeric_index/mutable_numeric_index.rs:249-301`

**Map 索引**：
```rust
/// Open mutable map index from RocksDB storage
pub fn open_rocksdb(
    db: Arc<RwLock<DB>>,
    field_name: &str,
    create_if_missing: bool,
) -> OperationResult<Option<Self>> {
    let store_cf_name = MapIndex::<N>::storage_cf_name(field_name);
    // Load in-memory index from RocksDB
    let mut map = HashMap::<_, RoaringBitmap>::new();
    let mut point_to_values = Vec::new();
    for (record, _) in db_wrapper.lock_db().iter()? {
        let (value, idx) = MapIndex::<N>::decode_db_record(record)?;
        // 重建内存索引
        // ...
    }
    // ...
}
```

**位置**: `lib/segment/src/index/field_index/map_index/mutable_map_index.rs:65-120`

#### 6.2.3 索引选择器

Qdrant 使用 `IndexSelector` 来根据存储类型选择索引实现：

```rust
pub enum IndexSelector<'a> {
    /// In-memory index on RocksDB, appendable or non-appendable
    #[cfg(feature = "rocksdb")]
    RocksDb(IndexSelectorRocksDb<'a>),
    /// On disk or in-memory index on mmaps, non-appendable
    Mmap(IndexSelectorMmap<'a>),
    /// In-memory index on gridstore, appendable
    Gridstore(IndexSelectorGridstore<'a>),
}

#[cfg(feature = "rocksdb")]
pub struct IndexSelectorRocksDb<'a> {
    pub db: &'a Arc<parking_lot::RwLock<rocksdb::DB>>,
    pub is_appendable: bool,
}
```

**位置**: `lib/segment/src/index/field_index/index_selector.rs:30-47`

#### 6.2.4 迁移策略

Qdrant 提供了从 RocksDB 索引迁移到 mmap 索引的机制：

```rust
// Actively migrate away from RocksDB indices
#[cfg(feature = "rocksdb")]
if common::flags::feature_flags().migrate_rocksdb_payload_indices
    && indexes.iter().any(|index| index.is_rocksdb())
{
    log::info!("Migrating away from RocksDB indices for field `{field}`");
    // 迁移逻辑...
}
```

**位置**: `lib/segment/src/index/struct_payload_index.rs:250-273`

### 6.3 存储对比

| 索引类型 | 早期版本（RocksDB） | 新版本（Mmap） |
|---------|-------------------|---------------|
| **HNSW 索引** | 独立文件（`graph.bin`, `links.bin`） | 独立文件（`graph.bin`, `links.bin`） |
| **Payload 索引** | RocksDB 列族 | Mmap 文件 |
| **存储位置** | RocksDB 数据库 | 文件系统 |
| **访问方式** | 通过 RocksDB API | 通过 mmap |
| **内存使用** | 加载到内存 | 按需加载 |

### 6.4 为什么 HNSW 索引不存储在 RocksDB 中？

1. **访问模式不匹配**：
   - HNSW 图需要高效的随机访问和遍历
   - RocksDB 的键值访问模式不适合图遍历

2. **性能考虑**：
   - 文件存储提供更好的顺序和随机访问性能
   - mmap 提供零拷贝访问

3. **独立性**：
   - HNSW 索引的存储与向量和 payload 的存储方式无关
   - 可以独立优化和扩展

---

## 7. 总结

Qdrant 中的索引存储采用以下策略：

1. **HNSW 索引**：
   - 支持内存存储和磁盘存储（mmap）两种模式
   - 通过 `on_disk` 配置项控制
   - 默认使用内存存储以获得最佳性能

2. **Payload 索引**：
   - **早期版本**：存储在 RocksDB 的列族中
   - **新版本**：统一使用 mmap 存储
   - 支持按需加载和预加载两种模式
   - 通过 `populate` 参数控制

3. **存储优化**：
   - 使用内存映射（mmap）实现零拷贝访问
   - 使用 `madvise` 优化内存访问模式
   - 支持多种压缩格式以节省存储空间

4. **迁移策略**：
   - Qdrant 正在逐步迁移远离 RocksDB
   - 提供了自动迁移机制
   - 新部署默认使用 mmap 存储

这种设计使得 Qdrant 能够在内存和磁盘之间灵活平衡，既保证了性能，又支持大规模数据的存储。
