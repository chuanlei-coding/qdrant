# Qdrant 的 GPU 优化说明

## 概述

Qdrant 的 GPU 优化主要用于**索引构建阶段**，通过 Vulkan API 在 GPU 上并行执行 HNSW 图构建操作，显著加速大规模向量索引的创建过程。

## GPU 优化的主要阶段

### 1. **索引构建阶段（Indexing）** ⚡ 主要提速阶段

GPU 优化在索引构建阶段发挥最大作用，主要体现在以下几个方面：

#### 1.1 HNSW 图构建

**位置**: `lib/segment/src/index/hnsw_index/gpu/gpu_graph_builder.rs`

**功能**: 在 GPU 上并行构建 HNSW 图结构

```rust
pub fn build_hnsw_on_gpu(
    gpu_insert_context: &mut GpuInsertContext,
    reference_graph: &GraphLayersBuilder,
    groups_count: usize,  // 并行处理的点数
    entry_points_num: usize,
    cpu_linked_points: usize,
    ids: Vec<PointOffsetType>,
    points_scorer_builder: impl Fn(PointOffsetType) -> OperationResult<FilteredScorer>,
    stopped: &AtomicBool,
) -> OperationResult<GraphLayersBuilder>
```

**提速效果**:
- 并行处理多个向量的插入操作（`groups_count` 个并行点）
- 默认支持 512 个并行组（可通过配置调整）
- 逐层构建图结构，每层都在 GPU 上并行处理

#### 1.2 逐层构建（Level-by-Level Building）

**位置**: `lib/segment/src/index/hnsw_index/gpu/gpu_level_builder.rs`

**功能**: 在 GPU 上逐层构建 HNSW 图的每一层

```rust
pub fn build_level_on_gpu(
    gpu_search_context: &mut GpuInsertContext,
    batched_points: &BatchedPoints,
    skip_count: usize,
    level: usize,
    stopped: &AtomicBool,
) -> OperationResult<()>
```

**提速效果**:
- 批量插入向量到图中（`gpu_batched_insert`）
- 批量更新入口点（`gpu_batched_update_entries`）
- 每层独立并行处理，充分利用 GPU 并行能力

#### 1.3 向量插入操作

**位置**: `lib/segment/src/index/hnsw_index/gpu/shaders/run_insert_vector.comp`

**功能**: GPU shader 执行向量插入和链接更新

**关键操作**:
- 贪心搜索找到最近邻
- 计算向量相似度
- 更新图的链接关系
- 执行启发式优化

**提速效果**:
- 在 GPU 上并行执行多个向量的插入操作
- 向量相似度计算在 GPU 上并行完成
- 链接更新操作并行化

### 2. **向量相似度计算** ⚡ 核心计算加速

**位置**: `lib/segment/src/index/hnsw_index/gpu/shaders/vector_storage_*.comp`

**功能**: 在 GPU shader 中计算向量相似度

**支持的相似度度量**:
- **Cosine 相似度** (`COSINE_DISTANCE`)
- **欧氏距离** (`EUCLID_DISTANCE`)
- **点积** (`DOT_DISTANCE`)
- **曼哈顿距离** (`MANHATTAN_DISTANCE`)

**支持的向量类型**:
- **Float32** (`vector_storage_f32.comp`)
- **Float16** (`vector_storage_f16.comp`) - 可选半精度加速
- **Uint8** (`vector_storage_u8.comp`)
- **量化向量**:
  - Product Quantization (PQ) - `vector_storage_pq.comp`
  - Scalar Quantization (SQ) - `vector_storage_sq.comp`
  - Binary Quantization (BQ) - `vector_storage_bq.comp`

**提速效果**:
- 向量相似度计算在 GPU 上并行执行
- 支持批量向量计算
- 半精度（f16）可进一步提升性能

### 3. **贪心搜索（Greedy Search）** ⚡ 搜索加速

**位置**: `lib/segment/src/index/hnsw_index/gpu/shaders/search_context.comp`

**功能**: 在 GPU 上执行贪心搜索，用于查找最近邻

```rust
pub fn greedy_search(
    &mut self,
    requests: &[GpuRequest],
    prev_results_count: usize,
) -> OperationResult<Vec<PointOffsetType>>
```

**提速效果**:
- 并行执行多个搜索请求
- 在 GPU 上执行图遍历和候选点评估
- 减少 CPU-GPU 数据传输

### 4. **向量存储和访问** ⚡ 数据传输优化

**位置**: `lib/segment/src/index/hnsw_index/gpu/gpu_vector_storage/mod.rs`

**功能**: 将向量数据上传到 GPU 内存，加速访问

**关键特性**:
- **批量上传**: 使用 staging buffer 批量上传向量
- **多存储支持**: 支持 4 个存储缓冲区（`STORAGES_COUNT = 4`）
- **内存对齐**: GPU 向量大小对齐到 subgroup 大小
- **分块上传**: 64MB 分块上传，避免内存溢出

**提速效果**:
- 向量数据常驻 GPU 内存，减少数据传输
- 支持多种向量格式（dense, multi, quantized）
- 优化内存访问模式

## GPU 配置

### 配置选项

在 `config/config.yaml` 中可以配置 GPU 选项：

```yaml
gpu:
  indexing: true                    # 启用 GPU 索引
  force_half_precision: false       # 强制使用半精度（f16）
  groups_count: 512                # 并行处理的点数（默认 512）
  device_filter: ""                 # GPU 设备过滤器
  devices: null                     # 指定使用的 GPU 设备索引
  parallel_indexes: 1              # 并行索引进程数
  allow_integrated: false          # 允许使用集成 GPU
  allow_emulated: false            # 允许使用模拟 GPU（用于 CI）
```

### 关键参数说明

1. **`groups_count`**: 
   - 控制 GPU 上并行处理的点数
   - 默认值 512，可根据 GPU 型号调整
   - 值越大，并行度越高，但需要更多 GPU 内存

2. **`force_half_precision`**:
   - 强制将 f32 向量转换为 f16 在 GPU 上处理
   - 可以减少 GPU 内存使用，提升性能
   - 仅在 GPU 内存中转换，不影响存储类型

3. **`parallel_indexes`**:
   - 允许同时运行的并行索引进程数
   - 默认 1，如果有多个 GPU 可以增加

## 提速效果总结

### 主要提速阶段

| 阶段 | 提速效果 | 说明 |
|------|---------|------|
| **索引构建** | ⭐⭐⭐⭐⭐ | 主要提速阶段，并行构建 HNSW 图 |
| **向量相似度计算** | ⭐⭐⭐⭐⭐ | 核心计算在 GPU 上并行执行 |
| **贪心搜索** | ⭐⭐⭐⭐ | 搜索操作在 GPU 上并行化 |
| **向量存储** | ⭐⭐⭐ | 减少数据传输，加速访问 |

### 适用场景

GPU 优化特别适合以下场景：

1. **大规模索引构建**:
   - 百万级以上的向量索引
   - 需要快速构建索引的场景

2. **高维向量**:
   - 高维向量（如 768、1024 维）
   - GPU 并行计算优势明显

3. **批量索引**:
   - 需要频繁重建索引
   - 增量索引更新

4. **性能敏感场景**:
   - 对索引构建速度要求高
   - 有 GPU 硬件资源

### 不适用场景

1. **小规模数据**:
   - 向量数量较少（< 10万）
   - GPU 初始化开销可能超过收益

2. **稀疏向量**:
   - 目前不支持稀疏向量的 GPU 加速
   - 稀疏向量仍使用 CPU 处理

3. **无 GPU 硬件**:
   - 需要支持 Vulkan 的 GPU
   - 集成 GPU 需要明确启用

## 技术实现

### GPU API

- **Vulkan API**: 使用 Vulkan 进行 GPU 计算
- **Compute Shaders**: 使用 GLSL compute shaders 执行计算
- **Subgroup Operations**: 利用 GPU subgroup 操作优化性能

### 关键组件

1. **`GpuVectorStorage`**: GPU 向量存储管理
2. **`GpuInsertContext`**: GPU 插入上下文管理
3. **`GpuGraphBuilder`**: GPU 图构建器
4. **`GpuLevelBuilder`**: GPU 层级构建器
5. **`GpuDevicesManager`**: GPU 设备管理器

### 代码位置

- **GPU 模块**: `lib/segment/src/index/hnsw_index/gpu/`
- **Shader 代码**: `lib/segment/src/index/hnsw_index/gpu/shaders/`
- **GPU 库**: `lib/gpu/`
- **配置**: `src/settings.rs` (GpuConfig)
- **初始化**: `src/main.rs` (GPU 设备初始化)

## 性能优化建议

1. **选择合适的 `groups_count`**:
   - 根据 GPU 型号和内存调整
   - 通常 512 是较好的默认值

2. **启用半精度**:
   - 如果 GPU 支持，启用 `force_half_precision`
   - 可以减少内存使用，提升性能

3. **多 GPU 支持**:
   - 如果有多个 GPU，可以增加 `parallel_indexes`
   - 使用 `devices` 指定使用的 GPU

4. **监控 GPU 使用**:
   - 观察 GPU 内存使用情况
   - 调整批次大小和并行度

## 总结

Qdrant 的 GPU 优化主要集中在**索引构建阶段**，通过并行化 HNSW 图构建、向量相似度计算和贪心搜索操作，显著加速大规模向量索引的创建。对于大规模、高维向量的索引构建场景，GPU 优化可以带来显著的性能提升。
