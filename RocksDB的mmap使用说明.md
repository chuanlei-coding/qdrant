# RocksDB 的 mmap 使用说明

## 概述

**RocksDB 本身支持 mmap（内存映射）技术**，但**默认情况下不启用**。RocksDB 提供了配置选项来启用 mmap 功能，但需要根据具体场景进行权衡。

## RocksDB 的 mmap 支持

### 1. **默认行为**

RocksDB **默认不使用 mmap**，而是使用标准的文件读写操作（`read()`/`write()` 系统调用）来访问数据文件。

**原因**:
- 标准 I/O 在大多数场景下性能更稳定
- mmap 可能带来内存管理和一致性方面的挑战
- 写入操作使用 mmap 的性能可能不如直接 I/O

### 2. **可选的 mmap 功能**

RocksDB 提供了两个配置选项来启用 mmap：

#### 2.1 `allow_mmap_reads`

**用途**: 允许使用 mmap 进行读取操作

**效果**:
- 将 SSTable（Sorted String Table）文件映射到内存空间
- 应用程序可以直接访问内存中的数据
- 避免 `read()` 系统调用的开销

**使用场景**:
- 读多写少的场景
- 数据文件较大且访问模式适合 mmap
- 系统有足够的内存来缓存映射的文件

#### 2.2 `allow_mmap_writes`

**用途**: 允许使用 mmap 进行写入操作

**注意**: 
- **性能可能不如直接 I/O**
- 早期实现中，使用 mmap 写入的性能并不理想
- 需要仔细评估和测试

### 3. **WAL（Write-Ahead Log）不使用 mmap**

RocksDB 的 **WAL 默认不使用 mmap**，原因：

1. **性能考虑**: 使用 mmap 写入 WAL 的性能可能不如直接 I/O
2. **一致性要求**: WAL 需要保证数据持久化，直接 I/O 更可靠
3. **设计选择**: RocksDB 选择了其他更高效的写入方式

## Qdrant 中 RocksDB 的配置

### 1. **Qdrant 的 RocksDB 配置**

在 Qdrant 中，RocksDB 的配置如下：

**位置**: `lib/segment/src/common/rocksdb_wrapper.rs`

```rust
pub fn make_db_options() -> Options {
    let mut options: Options = Options::default();
    options.set_write_buffer_size(DB_CACHE_SIZE); // 10 MB
    options.create_if_missing(true);
    options.set_log_level(LogLevel::Error);
    options.set_max_open_files(DB_MAX_OPEN_FILES as i32); // 256
    options.set_compression_type(rocksdb::DBCompressionType::Lz4);
    
    // Qdrant 依赖自己的 WAL 来保证持久性
    options.set_wal_recovery_mode(DBRecoveryMode::TolerateCorruptedTailRecords);
    
    // 注意：没有设置 allow_mmap_reads 或 allow_mmap_writes
    options
}
```

### 2. **Qdrant 未启用 mmap**

从代码可以看出，**Qdrant 在 RocksDB 配置中未启用 mmap 功能**：

- ❌ 没有设置 `allow_mmap_reads`
- ❌ 没有设置 `allow_mmap_writes`
- ✅ 使用 RocksDB 的默认 I/O 方式（标准文件读写）

**原因**:
1. **Qdrant 有自己的 WAL**: Qdrant 依赖自己的 WAL 来保证持久性，不依赖 RocksDB 的 WAL
2. **迁移计划**: Qdrant 正在从 RocksDB 迁移到 mmap 存储方案
3. **性能考虑**: 对于向量数据库场景，直接使用 mmap 可能比通过 RocksDB 的 mmap 更高效

## RocksDB mmap vs Qdrant mmap

### 1. **架构差异**

| 特性 | RocksDB mmap | Qdrant mmap |
|------|-------------|-------------|
| **架构层级** | 在 RocksDB 内部使用 | 在应用层直接使用 |
| **数据格式** | SSTable（LSM-Tree） | 原始向量数据 |
| **访问方式** | 通过 RocksDB API | 直接内存访问 |
| **控制粒度** | RocksDB 控制 | Qdrant 完全控制 |

### 2. **性能差异**

**RocksDB mmap**:
- 需要经过 RocksDB 的抽象层
- 需要处理 LSM-Tree 的复杂性
- 可能有多层缓存（RocksDB 缓存 + 操作系统页面缓存）

**Qdrant mmap**:
- 零拷贝直接访问
- 没有中间抽象层
- 可以针对向量访问模式优化（madvise、多 mmap 等）

### 3. **使用场景**

**RocksDB mmap 适合**:
- 使用 RocksDB 作为存储引擎
- 需要 RocksDB 的其他功能（事务、压缩等）
- 数据访问模式适合 LSM-Tree

**Qdrant mmap 适合**:
- 向量数据库场景
- 需要直接控制内存映射
- 需要针对向量访问模式优化
- 追求极致性能

## 为什么 Qdrant 不使用 RocksDB 的 mmap

### 1. **架构设计**

Qdrant 选择在应用层直接使用 mmap，而不是通过 RocksDB：

- **更直接的控制**: 可以针对向量访问模式进行优化
- **更少的抽象层**: 减少中间层的开销
- **更好的性能**: 零拷贝直接访问向量数据

### 2. **迁移策略**

Qdrant 正在从 RocksDB 迁移到直接使用 mmap：

- **逐步迁移**: 通过 feature flags 控制迁移
- **向后兼容**: 支持旧版本的 RocksDB 数据
- **性能提升**: 直接使用 mmap 可以获得更好的性能

### 3. **技术优势**

直接使用 mmap 的优势：

- **多 mmap 优化**: 可以根据访问模式创建多个 mmap（随机/顺序）
- **madvise 优化**: 可以针对向量访问模式设置 madvise 策略
- **异步 I/O**: 可以使用 io_uring 进行异步读取
- **类型安全**: 可以直接将 mmap 映射为 Rust 类型

## RocksDB mmap 的使用建议

### 1. **何时启用 RocksDB mmap**

如果使用 RocksDB 作为存储引擎，可以考虑启用 mmap：

**适合场景**:
- 读多写少的场景
- 数据文件较大
- 有足够的内存
- 访问模式适合 mmap

**启用方式**:
```rust
let mut options = Options::default();
options.set_allow_mmap_reads(true);  // 启用 mmap 读取
// options.set_allow_mmap_writes(true);  // 谨慎启用 mmap 写入
```

### 2. **注意事项**

**性能测试**:
- mmap 的性能效果因工作负载而异
- 需要在实际场景中进行测试和评估
- 可能在某些情况下导致性能下降

**内存管理**:
- mmap 会增加内存使用
- 需要监控内存使用情况
- 操作系统会自动管理页面缓存

**一致性考虑**:
- mmap 可能带来内存管理和一致性方面的挑战
- 需要确保数据一致性

## 总结

### RocksDB 的 mmap 支持

1. **支持但默认不启用**: RocksDB 支持 mmap，但默认使用标准 I/O
2. **可选配置**: 可以通过 `allow_mmap_reads` 和 `allow_mmap_writes` 启用
3. **主要用于读取**: mmap 主要用于读取 SSTable 文件
4. **WAL 不使用 mmap**: Write-Ahead Log 默认不使用 mmap

### Qdrant 的选择

1. **未启用 RocksDB mmap**: Qdrant 在 RocksDB 配置中未启用 mmap
2. **直接使用 mmap**: Qdrant 在应用层直接使用 mmap，而不是通过 RocksDB
3. **迁移策略**: Qdrant 正在从 RocksDB 迁移到直接使用 mmap 存储方案
4. **性能优势**: 直接使用 mmap 可以获得更好的性能和更灵活的控制

### 关键区别

- **RocksDB mmap**: 在 RocksDB 内部使用，需要经过 RocksDB 的抽象层
- **Qdrant mmap**: 在应用层直接使用，零拷贝直接访问，可以针对向量访问模式优化

Qdrant 选择直接使用 mmap 而不是通过 RocksDB，是为了获得更好的性能和控制能力，特别是在向量数据库这种对性能要求极高的场景中。
