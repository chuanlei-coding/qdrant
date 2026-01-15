# Qdrant 的 mmap 存储方案与 LSM 技术分析

## 概述

**Qdrant 的基于 mmap 的存储方案没有使用 LSM（Log-Structured Merge Tree）技术**。虽然某些组件（如 Gridstore）采用了类似 LSM 的追加写入模式，但整体架构与 LSM 树有本质区别。

## 核心结论

### 1. **向量存储：不使用 LSM**

Qdrant 的向量存储实现（`MmapDenseVectors`、`ChunkedMmapVectors`）**完全不使用 LSM 技术**：

- **直接存储**：向量数据直接存储在 mmap 文件中
- **顺序布局**：向量按顺序存储在文件中
- **分块存储**：使用分块（chunks）来支持动态扩展
- **无 LSM 特征**：没有多层级结构、没有合并操作、没有 SSTable

### 2. **Gridstore：部分类似但非 LSM**

Gridstore（用于 payload 存储）有一些**类似 LSM 的特征**，但**不是真正的 LSM 实现**：

- ✅ **追加写入**：更新不是原地更新，而是插入新值
- ❌ **无多层级结构**：LSM 的核心是多层级结构
- ❌ **无合并操作**：没有 compaction 机制
- ❌ **无排序 SSTable**：没有排序的字符串表

## 详细分析

### 1. **向量存储架构**

#### 1.1 不可变向量存储（MmapDenseVectors）

**位置**: `lib/segment/src/vector_storage/dense/mmap_dense_vectors.rs`

**数据布局**:
```
[HEADER: 4 bytes "data"]
[Vector 0: dim * sizeof(T) bytes]
[Vector 1: dim * sizeof(T) bytes]
...
[Vector N-1: dim * sizeof(T) bytes]
```

**特点**:
- **直接映射**：向量数据直接映射到内存
- **固定布局**：每个向量占用固定大小的空间
- **顺序存储**：向量按 ID 顺序存储
- **无 LSM 特征**：完全没有 LSM 树的结构

#### 1.2 分块向量存储（ChunkedMmapVectors）

**位置**: `lib/segment/src/vector_storage/chunked_mmap_vectors.rs`

**文件结构**:
```
vectors/
├── config.json          # 配置信息
├── status.dat           # 状态信息
├── chunk_0.dat         # 第 0 块
├── chunk_1.dat         # 第 1 块
└── ...
```

**特点**:
- **分块存储**：将向量数据分割成固定大小的块
- **动态扩展**：按需创建新块
- **直接访问**：通过块索引直接访问向量
- **无 LSM 特征**：没有层级结构、没有合并操作

### 2. **Gridstore 架构分析**

#### 2.1 Gridstore 的设计

**位置**: `lib/gridstore/src/gridstore.rs`

**核心组件**:
```rust
pub struct Gridstore<V> {
    config: StorageConfig,           // 配置
    tracker: Arc<RwLock<Tracker>>,   // 指针映射（PointOffset -> ValuePointer）
    pages: Arc<RwLock<Vec<Page>>>,   // 数据页（mmap）
    bitmask: Arc<RwLock<Bitmask>>,   // 块使用情况位图
    // ...
}
```

**文件结构**:
```
payload_storage/
├── config.json          # 配置
├── tracker.dat          # 指针映射文件
├── bitmask.dat          # 块使用情况位图
├── page_0.dat           # 数据页 0
├── page_1.dat           # 数据页 1
└── ...
```

#### 2.2 类似 LSM 的特征

**追加写入模式**:

从 `readme.md` 可以看出：
```
Updates:
  - not done in place, always a new value is inserted
  - calculation of the new regions gaps is done on the fly
  - the tracker is updated in-memory, only persisted on flush
```

**实现**:
```rust
pub fn put_value(
    &mut self,
    point_offset: PointOffset,
    value: &V,
    hw_counter: HwMetricRefCounter,
) -> Result<bool> {
    // 1. 压缩值
    let comp_value = self.compress(value_bytes);
    
    // 2. 找到可用的块（可能在新页面）
    let (start_page_id, block_offset) =
        self.find_or_create_available_blocks(required_blocks)?;
    
    // 3. 写入新位置（不是原地更新）
    self.write_into_pages(&comp_value, start_page_id, block_offset);
    
    // 4. 更新指针（在内存中）
    tracker_guard.set(
        point_offset,
        ValuePointer::new(start_page_id, block_offset, value_size as u32),
    );
    
    // 5. 旧数据在 flush 时释放
}
```

**特点**:
- ✅ **追加写入**：新值写入新位置，不覆盖旧值
- ✅ **延迟持久化**：Tracker 更新在内存中，flush 时持久化
- ✅ **旧数据回收**：旧数据的块在 flush 时标记为可用

#### 2.3 与 LSM 的区别

**LSM 树的核心特征**:

1. **多层级结构**：
   - L0（内存）：MemTable
   - L1, L2, ...（磁盘）：多个 SSTable 层级

2. **合并操作（Compaction）**：
   - 定期合并多个 SSTable
   - 删除旧数据和重复数据
   - 保持数据有序

3. **排序的 SSTable**：
   - 每个 SSTable 内部有序
   - 支持范围查询

4. **写入放大**：
   - 数据可能被多次写入（写入 MemTable，然后写入 L0，再合并到 L1...）

**Gridstore 的实现**:

1. **单层结构**：
   - 只有页面（Pages），没有多层级
   - 所有数据在同一层级

2. **无合并操作**：
   - 没有 compaction 机制
   - 旧数据通过 Bitmask 标记为可用，但不主动合并

3. **无序存储**：
   - 数据按块存储，不保证有序
   - 通过 Tracker 索引定位数据

4. **简单的空间回收**：
   - 旧数据的块标记为可用
   - 新写入时优先使用空闲块
   - 没有主动的合并和压缩

### 3. **Gridstore 的实际架构**

#### 3.1 数据组织方式

**分页存储**:
- **页面大小**：32 MB（默认）
- **块大小**：128 字节（默认）
- **区域大小**：8192 块（默认）

**数据定位**:
```
PointOffset (ID) 
    ↓ (通过 Tracker)
ValuePointer (page_id, block_offset, length)
    ↓ (通过 Page)
实际数据（压缩后的 payload）
```

#### 3.2 写入流程

```
1. put_value(point_id, payload)
   ↓
2. 压缩 payload (LZ4)
   ↓
3. 计算需要的块数
   ↓
4. 在 Bitmask 中查找可用块（或创建新页面）
   ↓
5. 写入数据到页面（mmap）
   ↓
6. 更新 Bitmask（标记块为已使用）
   ↓
7. 更新 Tracker（在内存中，point_id -> ValuePointer）
   ↓
8. flush() 时：
   - 持久化 Tracker 更新
   - 释放旧数据的块（在 Bitmask 中标记为可用）
   - 刷新所有 mmap
```

#### 3.3 读取流程

```
1. get_value(point_id)
   ↓
2. 从 Tracker 获取 ValuePointer（优先从 pending_updates）
   ↓
3. 根据 ValuePointer 从页面读取数据
   ↓
4. 解压缩数据
   ↓
5. 反序列化为 Payload
```

### 4. **与 LSM 的对比**

| 特性 | LSM 树 | Gridstore | Mmap 向量存储 |
|------|--------|-----------|---------------|
| **多层级结构** | ✅ 是 | ❌ 否 | ❌ 否 |
| **合并操作** | ✅ 是（Compaction） | ❌ 否 | ❌ 否 |
| **排序存储** | ✅ 是（SSTable 有序） | ❌ 否 | ❌ 否（按 ID 顺序） |
| **追加写入** | ✅ 是 | ✅ 是 | ✅ 是（ChunkedMmap） |
| **写入放大** | ✅ 是（多次写入） | ⚠️ 部分（旧数据不立即删除） | ❌ 否 |
| **范围查询** | ✅ 高效 | ❌ 不支持 | ❌ 不支持 |
| **索引结构** | ✅ B-Tree 索引 | ✅ Tracker（指针映射） | ❌ 直接访问 |
| **数据压缩** | ✅ 支持 | ✅ LZ4 | ❌ 不支持 |

### 5. **为什么 Gridstore 不使用 LSM**

#### 5.1 设计目标不同

**LSM 树的目标**:
- 优化写入性能（追加写入）
- 支持范围查询
- 处理大规模数据

**Gridstore 的目标**:
- 存储可变长度的 payload
- 支持随机访问（通过 ID）
- 使用 mmap 实现零拷贝
- 简单的空间管理

#### 5.2 访问模式不同

**LSM 适合**:
- 写多读少的场景
- 需要范围查询
- 数据有序存储

**Gridstore 适合**:
- 通过 ID 随机访问
- 不需要范围查询
- 数据无序存储

#### 5.3 实现复杂度

**LSM 树**:
- 需要实现多层级管理
- 需要实现合并操作
- 需要处理写入放大
- 复杂度较高

**Gridstore**:
- 简单的分页存储
- 通过 Tracker 索引定位
- 通过 Bitmask 管理空间
- 复杂度较低

### 6. **Gridstore 的优势**

#### 6.1 简单高效

- **无合并开销**：不需要定期合并操作
- **直接访问**：通过 Tracker 直接定位数据
- **mmap 优势**：零拷贝访问，操作系统管理缓存

#### 6.2 适合向量数据库场景

- **随机访问**：向量数据库主要按 ID 访问
- **不需要范围查询**：payload 查询通常通过 ID
- **写入频率低**：向量数据库主要是读多写少

#### 6.3 内存管理灵活

- **按需加载**：只加载访问的页面
- **预加载支持**：可以预加载所有页面
- **操作系统管理**：由操作系统管理页面缓存

## 总结

### 核心结论

1. **向量存储不使用 LSM**：
   - `MmapDenseVectors` 和 `ChunkedMmapVectors` 完全不使用 LSM
   - 是简单的顺序存储或分块存储

2. **Gridstore 部分类似但非 LSM**：
   - 采用追加写入模式（类似 LSM）
   - 但没有多层级结构、合并操作、排序 SSTable
   - 更像是一个简单的追加写入存储 + 索引

3. **设计选择**：
   - Qdrant 选择简单高效的存储方案
   - 针对向量数据库的访问模式优化
   - 避免 LSM 的复杂性和开销

### 技术对比

| 存储方案 | LSM 使用 | 架构特点 |
|---------|---------|---------|
| **MmapDenseVectors** | ❌ 不使用 | 直接 mmap，顺序存储 |
| **ChunkedMmapVectors** | ❌ 不使用 | 分块 mmap，动态扩展 |
| **Gridstore** | ⚠️ 部分类似 | 追加写入，但无 LSM 结构 |
| **RocksDB** | ✅ 使用 | 完整的 LSM 树实现 |

### 为什么选择不使用 LSM

1. **访问模式匹配**：
   - 向量数据库主要是随机访问（按 ID）
   - 不需要范围查询
   - LSM 的优势无法发挥

2. **性能考虑**：
   - 直接 mmap 访问更高效
   - 避免 LSM 的写入放大
   - 避免合并操作的开销

3. **实现简单**：
   - 简单的存储方案更容易维护
   - 更容易优化和调试
   - 更适合向量数据库场景

Qdrant 的 mmap 存储方案是一个**针对向量数据库优化的简单高效存储方案**，而不是基于 LSM 的复杂存储引擎。
