# Qdrant 基于 mmap 的存储方案技术剖析

## 概述

Qdrant 使用**内存映射文件（Memory-Mapped Files, mmap）**作为主要的持久化存储方案，用于替代 RocksDB。mmap 存储方案提供了高性能、灵活的内存管理和良好的可扩展性。

## 核心设计理念

### 1. **零拷贝访问**

mmap 允许直接将文件内容映射到进程的虚拟地址空间，实现零拷贝的数据访问：

- **直接内存访问**：数据在内存中，无需额外的序列化/反序列化
- **操作系统管理**：由操作系统负责页面缓存和内存管理
- **透明分页**：按需加载页面，自动处理内存不足的情况

### 2. **灵活的内存管理**

支持多种内存策略：

- **按需加载**：只在访问时加载页面到内存
- **预加载（Populate）**：启动时将所有页面加载到内存
- **混合策略**：根据访问模式动态调整

## 存储类型架构

### 1. **向量存储类型**

Qdrant 提供了多种 mmap 向量存储类型：

#### 1.1 不可变存储（Immutable）

**`MemmapDenseVectorStorage`** - 不可变的密集向量存储

**位置**: `lib/segment/src/vector_storage/dense/memmap_dense_vector_storage.rs`

**特点**:
- 不支持插入新向量
- 只能标记向量为删除
- 适合只读或很少更新的场景
- 使用单个 mmap 文件存储所有向量

**文件结构**:
```
vectors/
├── matrix.dat    # 向量数据文件（mmap）
└── deleted.dat   # 删除标记文件（mmap BitSlice）
```

**数据布局**:
```rust
// matrix.dat 布局
[HEADER: 4 bytes "data"]
[Vector 0: dim * sizeof(T) bytes]
[Vector 1: dim * sizeof(T) bytes]
...
[Vector N-1: dim * sizeof(T) bytes]

// deleted.dat 布局
[HEADER: 4 bytes "drop"]
[对齐填充]
[BitSlice: 每个 bit 表示一个向量的删除状态]
```

#### 1.2 可追加存储（Appendable）

**`AppendableMmapDenseVectorStorage`** - 可追加的密集向量存储

**位置**: `lib/segment/src/vector_storage/dense/appendable_dense_vector_storage.rs`

**特点**:
- 支持动态插入新向量
- 使用分块存储（ChunkedMmap）
- 支持大规模数据增长
- 适合频繁写入的场景

**实现**:
```rust
pub struct AppendableMmapDenseVectorStorage<T, S: ChunkedVectorStorage<T>> {
    vectors: S,              // 分块向量存储
    deleted: BitvecFlags,    // 动态删除标记
    distance: Distance,
    deleted_count: usize,
}
```

#### 1.3 分块存储（ChunkedMmap）

**`ChunkedMmapVectors`** - 分块 mmap 向量存储

**位置**: `lib/segment/src/vector_storage/chunked_mmap_vectors.rs`

**设计**:
- **分块策略**：将向量数据分割成固定大小的块（chunks）
- **动态扩展**：按需创建新块，支持无限增长
- **独立文件**：每个块存储在独立的文件中

**配置**:
```rust
struct ChunkedMmapConfig {
    chunk_size_bytes: usize,      // 块大小（字节）
    chunk_size_vectors: usize,   // 每块向量数量
    dim: usize,                   // 向量维度
    populate: Option<bool>,       // 是否预加载
}
```

**文件结构**:
```
vectors/
├── config.json          # 配置信息
├── status.dat           # 状态信息（当前向量数量）
├── chunk_0.dat         # 第 0 块
├── chunk_1.dat         # 第 1 块
└── ...
```

**块大小计算**:
```rust
const CHUNK_SIZE: usize = 64 * 1024; // 64 KB 默认块大小

let vector_size_bytes = dim * std::mem::size_of::<T>();
let chunk_size_vectors = CHUNK_SIZE / vector_size_bytes;
let corrected_chunk_size_bytes = chunk_size_vectors * vector_size_bytes;
```

**优势**:
- **增量扩展**：不需要重新分配整个文件
- **并行访问**：不同块可以并行读取
- **内存效率**：只加载需要的块到内存

### 2. **Payload 存储**

**`MmapPayloadStorage`** - 基于 Gridstore 的 mmap payload 存储

**位置**: `lib/segment/src/payload_storage/mmap_payload_storage.rs`

**实现**:
```rust
pub struct MmapPayloadStorage {
    storage: Gridstore<Payload>,  // Gridstore 提供底层存储
    populate: bool,                // 是否预加载
}
```

**Gridstore 设计**:
- **分页存储**：将 payload 数据分页存储
- **可变长度**：支持不同大小的 payload
- **高效查找**：通过索引快速定位数据

**文件结构**:
```
payload_storage/
├── pages/              # 数据页目录
│   ├── page_0.dat
│   ├── page_1.dat
│   └── ...
└── index/              # 索引文件
```

### 3. **量化向量存储**

**`QuantizedMmapStorage`** - 量化向量的 mmap 存储

**位置**: `lib/segment/src/vector_storage/quantized/quantized_mmap_storage.rs`

**特点**:
- 存储量化后的向量（PQ、SQ、BQ）
- 减少存储空间和内存使用
- 支持分块存储（`QuantizedChunkedMmapStorage`）

## 核心技术实现

### 1. **类型安全的内存映射**

Qdrant 实现了类型安全的内存映射抽象，直接将 mmap 数据转换为 Rust 类型：

**`MmapType<T>`** - 类型化的 mmap

**位置**: `lib/common/memory/src/mmap_type.rs`

**实现**:
```rust
pub struct MmapType<T: ?Sized + 'static> {
    r#type: &'static mut T,  // 类型化的引用
    mmap: Arc<MmapMut>,      // 底层 mmap
}

impl<T> Deref for MmapType<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.r#type
    }
}
```

**安全保证**:
- **对齐检查**：确保数据正确对齐
- **大小验证**：验证 mmap 大小匹配类型大小
- **生命周期管理**：通过 `Arc` 管理 mmap 生命周期

**使用示例**:
```rust
// 将 mmap 转换为 Status 类型
let status_mmap = open_write_mmap(&status_file, ...)?;
let status: MmapType<Status> = unsafe { MmapType::from(status_mmap) };

// 直接使用，就像普通类型
status.len = 1000;
```

### 2. **切片类型映射**

**`MmapSlice<T>`** - mmap 切片类型

**实现**:
```rust
pub struct MmapSlice<T: 'static> {
    slice: &'static [T],
    mmap: Arc<MmapMut>,
}
```

**用途**:
- 将 mmap 数据映射为 `&[T]` 切片
- 支持向量数组的直接访问
- 零拷贝的向量读取

### 3. **BitSlice 映射**

**`MmapBitSlice`** - mmap BitSlice 类型

**用途**:
- 存储删除标记（deleted flags）
- 高效的位操作
- 内存对齐优化

**实现**:
```rust
// 删除标记的内存布局
const HEADER_SIZE: usize = 4;
const fn deleted_mmap_data_start() -> usize {
    let align = mem::align_of::<usize>();
    HEADER_SIZE.div_ceil(align) * align  // 对齐到 usize 边界
}
```

### 4. **类型转换（Transmute）**

Qdrant 使用安全的类型转换函数：

**位置**: `lib/common/memory/src/mmap_ops.rs`

**实现**:
```rust
pub fn transmute_from_u8_to_slice<T>(data: &[u8]) -> &[T] {
    // 对齐检查
    debug_assert_eq!(
        data.as_ptr().align_offset(align_of::<T>()),
        0,
        "misaligned"
    );
    
    let len = data.len() / size_of::<T>();
    let ptr = data.as_ptr().cast::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
```

**安全保证**:
- **对齐验证**：确保数据正确对齐
- **大小验证**：确保字节数匹配类型大小
- **调试断言**：在 debug 模式下进行严格检查

## 性能优化技术

### 1. **madvise 优化**

**madvise** 系统调用告诉操作系统内存访问模式，优化页面缓存策略。

**位置**: `lib/common/memory/src/madvise.rs`

**支持的策略**:

#### 1.1 Random（随机访问）
```rust
Advice::Random  // MADV_RANDOM
```
- **用途**：随机访问模式
- **优化**：操作系统不预读，减少不必要的 I/O
- **场景**：HNSW 图搜索、随机向量访问

#### 1.2 Sequential（顺序访问）
```rust
Advice::Sequential  // MADV_SEQUENTIAL
```
- **用途**：顺序访问模式
- **优化**：操作系统预读后续页面
- **场景**：批量向量读取、顺序扫描

#### 1.3 Normal（正常访问）
```rust
Advice::Normal  // MADV_NORMAL
```
- **用途**：默认访问模式
- **优化**：操作系统使用默认策略
- **场景**：混合访问模式

**全局配置**:
```rust
// 设置全局 madvise 策略
memory::madvise::set_global(Advice::Random);

// 或在创建 mmap 时指定
let mmap = open_read_mmap(
    path,
    AdviceSetting::Advice(Advice::Sequential),
    false
)?;
```

### 2. **多 mmap 优化**

**多 mmap 支持**：对同一文件创建多个 mmap，使用不同的访问建议。

**位置**: `lib/segment/src/vector_storage/dense/mmap_dense_vectors.rs`

**实现**:
```rust
pub struct MmapDenseVectors<T> {
    mmap: Arc<Mmap>,              // 主 mmap（随机访问）
    _mmap_seq: Option<Arc<Mmap>>, // 顺序访问 mmap（可选）
    // ...
}

// 根据访问模式选择 mmap
fn raw_vector_offset<P: AccessPattern>(&self, offset: usize) -> &[T] {
    let mmap = if P::IS_SEQUENTIAL {
        &self.mmap_seq()  // 使用顺序访问 mmap
    } else {
        &self.mmap        // 使用随机访问 mmap
    };
    // ...
}
```

**优势**:
- **访问模式优化**：根据访问模式选择最优的 mmap
- **操作系统提示**：不同的 madvise 策略优化页面缓存
- **性能提升**：在混合访问模式下显著提升性能

**环境检测**:
```rust
// 检测是否支持多 mmap
pub static MULTI_MMAP_IS_SUPPORTED: LazyLock<bool> = ...;

// 某些环境（如 Docker on Windows）不支持多 mmap
// 可以通过环境变量禁用
QDRANT_NO_MULTI_MMAP=1
```

### 3. **页面预加载（Populate）**

**Populate** 功能在创建 mmap 时将所有页面加载到内存。

**实现**:
```rust
impl Madviseable for memmap2::Mmap {
    fn populate(&self) {
        #[cfg(target_os = "linux")]
        if *POPULATE_READ_IS_SUPPORTED {
            // 使用 MADV_POPULATE_READ（Linux 5.14+）
            match self.advise(memmap2::Advice::PopulateRead) {
                Ok(()) => return,
                Err(err) => log::warn!("Failed to populate: {err}"),
            }
        }
        // 回退到简单方法：读取每 512 字节
        populate_simple(self);
    }
}
```

**两种模式**:

#### 3.1 InRamMmap（内存模式）
```rust
PayloadStorageType::InRamMmap
VectorStorageType::InRamChunkedMmap
```
- **特点**：启动时预加载所有页面到内存
- **优势**：避免冷启动时的磁盘 I/O
- **代价**：占用更多内存

#### 3.2 Mmap（磁盘模式）
```rust
PayloadStorageType::Mmap
VectorStorageType::ChunkedMmap
```
- **特点**：按需加载页面
- **优势**：内存使用灵活
- **代价**：首次访问可能有磁盘 I/O

### 4. **异步 I/O（io_uring）**

**Linux io_uring** 支持异步向量读取。

**位置**: `lib/segment/src/vector_storage/dense/mmap_dense_vectors.rs`

**实现**:
```rust
#[cfg(target_os = "linux")]
use crate::vector_storage::async_io::UringReader;

pub struct MmapDenseVectors<T> {
    uring_reader: Mutex<Option<UringReader<T>>>,
    // ...
}

pub fn read_vectors_async(
    &self,
    points: impl Iterator<Item = PointOffsetType>,
    callback: impl FnMut(usize, PointOffsetType, &[T]),
) -> OperationResult<()> {
    #[cfg(target_os = "linux")]
    {
        self.process_points_uring(points, callback)
    }
    #[cfg(not(target_os = "linux"))]
    {
        self.process_points_simple(points, callback);
        Ok(())
    }
}
```

**优势**:
- **非阻塞 I/O**：异步读取，不阻塞线程
- **批量操作**：可以批量提交多个读取请求
- **性能提升**：在高并发场景下显著提升性能

### 5. **访问模式优化**

**访问模式检测**：根据访问模式选择最优的读取策略。

**实现**:
```rust
pub fn get_vectors<'a>(
    &'a self,
    keys: &[PointOffsetType],
    vectors: &'a mut [MaybeUninit<&'a [T]>],
) -> &'a [&'a [T]] {
    if is_read_with_prefetch_efficient_points(keys) {
        // 顺序访问：使用顺序 mmap
        let iter = keys.iter().map(|key| self.get_vector::<Sequential>(*key));
        maybe_uninit_fill_from(vectors, iter).0
    } else {
        // 随机访问：使用随机 mmap
        let iter = keys.iter().map(|key| self.get_vector::<Random>(*key));
        maybe_uninit_fill_from(vectors, iter).0
    }
}
```

**判断逻辑**:
- **顺序访问**：如果访问的向量 ID 是连续的或接近连续的
- **随机访问**：如果访问的向量 ID 是分散的

### 6. **批量读取优化**

**批量读取**：一次读取多个向量，减少系统调用。

**实现**:
```rust
const VECTOR_READ_BATCH_SIZE: usize = ...; // 批量大小

pub fn get_batch<'a>(
    &'a self,
    keys: &[VectorOffsetType],
    vectors: &'a mut [MaybeUninit<&'a [T]>],
) -> &'a [&'a [T]]
```

**优势**:
- **减少开销**：减少函数调用和边界检查
- **缓存友好**：提高 CPU 缓存命中率
- **预取优化**：可以触发硬件预取

## 数据布局和文件结构

### 1. **向量数据文件布局**

#### 1.1 不可变向量文件（matrix.dat）

```
[HEADER: 4 bytes "data"]
[Vector 0: dim * sizeof(T) bytes]
[Vector 1: dim * sizeof(T) bytes]
...
[Vector N-1: dim * sizeof(T) bytes]
```

**特点**:
- 固定大小，不支持追加
- 连续存储，访问效率高
- 适合只读场景

#### 1.2 分块向量文件（chunked）

```
vectors/
├── config.json          # JSON 配置
│   {
│     "chunk_size_bytes": 65536,
│     "chunk_size_vectors": 131,
│     "dim": 500,
│     "populate": false
│   }
├── status.dat           # 状态信息（二进制）
│   [len: usize]         # 当前向量数量
├── chunk_0.dat          # 第 0 块
│   [Vector 0..130]
├── chunk_1.dat          # 第 1 块
│   [Vector 131..261]
└── ...
```

**块大小计算**:
```rust
const CHUNK_SIZE: usize = 64 * 1024; // 64 KB

let vector_size = dim * size_of::<T>();
let vectors_per_chunk = CHUNK_SIZE / vector_size;
let actual_chunk_size = vectors_per_chunk * vector_size;
```

### 2. **删除标记文件布局**

```
[HEADER: 4 bytes "drop"]
[对齐填充: 对齐到 usize 边界]
[BitSlice: 每个 bit 表示一个向量的删除状态]
```

**对齐计算**:
```rust
const fn deleted_mmap_data_start() -> usize {
    let align = mem::align_of::<usize>();
    HEADER_SIZE.div_ceil(align) * align
}
```

**大小计算**:
```rust
fn deleted_mmap_size(num_vectors: usize) -> usize {
    let unit_size = mem::size_of::<usize>();
    let num_bytes = num_vectors.div_ceil(8);
    let num_usizes = num_bytes.div_ceil(unit_size);
    let data_size = num_usizes * unit_size;
    deleted_mmap_data_start() + data_size
}
```

### 3. **Payload 存储文件布局**

Gridstore 使用分页存储：

```
payload_storage/
├── pages/
│   ├── page_0.dat       # 数据页 0
│   ├── page_1.dat       # 数据页 1
│   └── ...
└── index/
    └── ...              # 索引文件
```

## 内存管理策略

### 1. **页面缓存管理**

**操作系统页面缓存**：
- mmap 数据由操作系统页面缓存管理
- 访问时自动加载到内存
- 内存不足时自动换出

**清除缓存**：
```rust
pub fn clear_cache(&self) -> OperationResult<()> {
    clear_disk_cache(&self.vectors_path)?;
    clear_disk_cache(&self.deleted_path)?;
    Ok(())
}
```

### 2. **内存对齐**

**对齐要求**：
- 数据必须对齐到类型对齐要求
- 使用 `align_of` 计算对齐
- 在 debug 模式下验证对齐

**对齐计算**:
```rust
fn deleted_mmap_data_start() -> usize {
    let align = mem::align_of::<usize>();
    HEADER_SIZE.div_ceil(align) * align
}
```

### 3. **内存映射生命周期**

**Arc 管理**：
```rust
pub struct MmapType<T> {
    r#type: &'static mut T,
    mmap: Arc<MmapMut>,  // 使用 Arc 共享所有权
}
```

**优势**:
- 多个引用可以共享同一个 mmap
- 自动管理生命周期
- 避免重复映射

## 与 RocksDB 的对比

### 1. **性能对比**

| 特性 | Mmap | RocksDB |
|------|------|---------|
| **读取性能** | ⭐⭐⭐⭐⭐ 零拷贝，直接内存访问 | ⭐⭐⭐ 需要序列化/反序列化 |
| **写入性能** | ⭐⭐⭐⭐ 直接写入内存 | ⭐⭐⭐⭐ 写入 WAL 和 MemTable |
| **内存使用** | ⭐⭐⭐⭐⭐ 操作系统管理，灵活 | ⭐⭐⭐ 固定缓存大小 |
| **扩展性** | ⭐⭐⭐⭐⭐ 分块存储，无限扩展 | ⭐⭐⭐⭐ 需要配置调优 |

### 2. **功能对比**

| 功能 | Mmap | RocksDB |
|------|------|---------|
| **追加写入** | ✅ 支持（ChunkedMmap） | ✅ 支持 |
| **随机读取** | ✅ 支持，性能优秀 | ✅ 支持 |
| **顺序读取** | ✅ 支持，有优化 | ✅ 支持 |
| **删除标记** | ✅ BitSlice，高效 | ✅ 支持 |
| **事务支持** | ❌ 不支持 | ✅ 支持 |
| **压缩** | ❌ 不支持 | ✅ 支持（LZ4） |

### 3. **适用场景**

**Mmap 适合**:
- 大规模向量存储
- 读多写少的场景
- 需要灵活内存管理的场景
- 追求极致性能的场景

**RocksDB 适合**:
- 需要事务支持
- 需要压缩
- 复杂的查询需求
- 小规模数据

## 最佳实践

### 1. **存储类型选择**

```rust
// 只读场景
VectorStorageType::Mmap

// 需要追加写入
VectorStorageType::ChunkedMmap

// 需要全部加载到内存
VectorStorageType::InRamChunkedMmap
```

### 2. **madvise 策略配置**

```rust
// 随机访问为主（如 HNSW 搜索）
memory::madvise::set_global(Advice::Random);

// 顺序访问为主（如批量处理）
memory::madvise::set_global(Advice::Sequential);
```

### 3. **内存管理**

```rust
// 预加载所有页面（启动时）
storage.populate()?;

// 清除页面缓存（释放内存）
storage.clear_cache()?;
```

### 4. **分块大小调优**

```rust
// 根据向量大小调整块大小
// 默认 64 KB，可以根据实际情况调整
const CHUNK_SIZE: usize = 128 * 1024; // 128 KB
```

## 技术挑战和解决方案

### 1. **多 mmap 兼容性**

**问题**：某些环境不支持多 mmap（如 Docker on Windows）

**解决方案**：
- 启动时检测多 mmap 支持
- 不支持时回退到单 mmap
- 通过环境变量禁用

### 2. **内存对齐**

**问题**：类型转换需要严格的内存对齐

**解决方案**：
- 在 debug 模式下验证对齐
- 使用对齐填充
- 提供安全的转换函数

### 3. **页面预加载**

**问题**：不同操作系统支持不同的预加载方式

**解决方案**：
- Linux 5.14+ 使用 `MADV_POPULATE_READ`
- 旧版本使用简单的读取方法
- 提供统一的接口

### 4. **并发安全**

**问题**：多个线程同时访问 mmap

**解决方案**：
- 使用 `Arc` 共享 mmap
- 只读访问不需要锁
- 写入操作使用适当的同步机制

## 总结

Qdrant 的 mmap 存储方案是一个**高性能、灵活、可扩展**的存储解决方案：

1. **零拷贝访问**：直接内存访问，无需序列化
2. **灵活的内存管理**：支持按需加载和预加载
3. **多种存储类型**：不可变、可追加、分块存储
4. **性能优化**：madvise、多 mmap、异步 I/O
5. **类型安全**：类型化的 mmap 抽象
6. **跨平台支持**：适配不同操作系统的特性

mmap 存储方案是 Qdrant 从 RocksDB 迁移的核心技术，为大规模向量数据库提供了坚实的基础。
