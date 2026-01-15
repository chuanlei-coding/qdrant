# Qdrant 支持的算子详解

## 概述

Qdrant 提供了丰富的算子（操作符）系统，用于过滤、查询和评分。这些算子可以分为以下几个主要类别：

1. **逻辑组合算子**：用于组合多个条件
2. **字段条件算子**：用于字段值的匹配和比较
3. **数学表达式算子**：用于计算和评分
4. **地理空间算子**：用于地理位置查询
5. **时间范围算子**：用于时间相关的查询
6. **特殊条件算子**：用于检查字段状态

---

## 一、逻辑组合算子

### 1.1 Filter 结构

Filter 是 Qdrant 中过滤条件的容器，支持以下逻辑组合：

```json
{
  "must": [...],      // 所有条件必须匹配（AND）
  "should": [...],    // 至少一个条件匹配（OR）
  "must_not": [...],  // 所有条件必须不匹配（NOT）
  "min_should": {     // 至少 N 个条件匹配
    "conditions": [...],
    "min_count": N
  }
}
```

#### must（必须匹配）
- **逻辑**：AND（与）
- **说明**：所有条件都必须满足
- **示例**：
```json
{
  "must": [
    {"key": "category", "match": {"value": "electronics"}},
    {"key": "price", "range": {"gte": 100, "lte": 1000}}
  ]
}
```

#### should（应该匹配）
- **逻辑**：OR（或）
- **说明**：至少一个条件满足即可
- **示例**：
```json
{
  "should": [
    {"key": "category", "match": {"value": "electronics"}},
    {"key": "category", "match": {"value": "computers"}}
  ]
}
```

#### must_not（必须不匹配）
- **逻辑**：NOT（非）
- **说明**：所有条件都必须不满足
- **示例**：
```json
{
  "must_not": [
    {"key": "status", "match": {"value": "out_of_stock"}}
  ]
}
```

#### min_should（最小匹配数）
- **逻辑**：至少 N 个条件匹配
- **说明**：在 should 条件中，至少满足 min_count 个条件
- **示例**：
```json
{
  "min_should": {
    "conditions": [
      {"key": "tag", "match": {"value": "featured"}},
      {"key": "tag", "match": {"value": "popular"}},
      {"key": "tag", "match": {"value": "new"}}
    ],
    "min_count": 2
  }
}
```

---

## 二、字段条件算子（FieldCondition）

### 2.1 Match（匹配）

用于匹配字段值，支持多种匹配模式：

#### keyword（关键字匹配）
- **类型**：精确匹配字符串关键字
- **示例**：
```json
{
  "key": "category",
  "match": {
    "keyword": "electronics"
  }
}
```

#### integer（整数匹配）
- **类型**：精确匹配整数值
- **示例**：
```json
{
  "key": "user_id",
  "match": {
    "integer": 12345
  }
}
```

#### boolean（布尔匹配）
- **类型**：匹配布尔值
- **示例**：
```json
{
  "key": "is_active",
  "match": {
    "boolean": true
  }
}
```

#### text（全文匹配）
- **类型**：全文搜索匹配
- **说明**：支持全文索引，进行文本搜索
- **示例**：
```json
{
  "key": "description",
  "match": {
    "text": "laptop computer"
  }
}
```

#### keywords（多关键字匹配）
- **类型**：匹配多个关键字（OR）
- **示例**：
```json
{
  "key": "tags",
  "match": {
    "keywords": ["electronics", "computers", "gadgets"]
  }
}
```

#### integers（多整数匹配）
- **类型**：匹配多个整数值（OR）
- **示例**：
```json
{
  "key": "user_ids",
  "match": {
    "integers": [1, 2, 3, 4, 5]
  }
}
```

#### except_integers（排除整数）
- **类型**：排除指定的整数值
- **示例**：
```json
{
  "key": "user_id",
  "match": {
    "except_integers": [999, 1000]
  }
}
```

#### except_keywords（排除关键字）
- **类型**：排除指定的关键字
- **示例**：
```json
{
  "key": "category",
  "match": {
    "except_keywords": ["deprecated", "archived"]
  }
}
```

#### phrase（短语匹配）
- **类型**：精确短语匹配
- **说明**：匹配完整的短语
- **示例**：
```json
{
  "key": "title",
  "match": {
    "phrase": "laptop computer"
  }
}
```

#### text_any（任意词匹配）
- **类型**：匹配文本中的任意词
- **说明**：只要文本中包含任意一个词即可
- **示例**：
```json
{
  "key": "description",
  "match": {
    "text_any": "laptop computer"
  }
}
```

### 2.2 Range（范围）

用于数值范围查询：

```json
{
  "key": "price",
  "range": {
    "gt": 100,   // 大于
    "gte": 100,  // 大于等于
    "lt": 1000,  // 小于
    "lte": 1000  // 小于等于
  }
}
```

**支持的比较操作符**：
- `gt`：大于（Greater Than）
- `gte`：大于等于（Greater Than or Equal）
- `lt`：小于（Less Than）
- `lte`：小于等于（Less Than or Equal）

**示例**：
```json
// 价格在 100 到 1000 之间
{
  "key": "price",
  "range": {
    "gte": 100,
    "lte": 1000
  }
}

// 评分大于 4.5
{
  "key": "rating",
  "range": {
    "gt": 4.5
  }
}
```

### 2.3 ValuesCount（值计数）

检查字段值的数量：

```json
{
  "key": "tags",
  "values_count": {
    "gt": 3,    // 值的数量大于 3
    "gte": 2,   // 值的数量大于等于 2
    "lt": 10,   // 值的数量小于 10
    "lte": 5    // 值的数量小于等于 5
  }
}
```

**用途**：检查数组字段的元素数量

**示例**：
```json
// 标签数量在 2 到 5 之间
{
  "key": "tags",
  "values_count": {
    "gte": 2,
    "lte": 5
  }
}
```

### 2.4 is_empty（空值检查）

检查字段是否为空：

```json
{
  "key": "description",
  "is_empty": true  // 字段为空或不存在
}
```

### 2.5 is_null（NULL 值检查）

检查字段是否为 NULL：

```json
{
  "key": "optional_field",
  "is_null": true  // 字段值为 NULL
}
```

---

## 三、地理空间算子

### 3.1 GeoBoundingBox（地理边界框）

查询指定矩形区域内的点：

```json
{
  "key": "location",
  "geo_bounding_box": {
    "top_left": {
      "lon": -122.5,  // 左上角经度
      "lat": 37.8     // 左上角纬度
    },
    "bottom_right": {
      "lon": -122.3,  // 右下角经度
      "lat": 37.6     // 右下角纬度
    }
  }
}
```

**用途**：查询矩形区域内的地理位置

### 3.2 GeoRadius（地理半径）

查询指定圆心和半径内的点：

```json
{
  "key": "location",
  "geo_radius": {
    "center": {
      "lon": -122.4,
      "lat": 37.7
    },
    "radius": 1000  // 半径（米）
  }
}
```

**用途**：查询圆形区域内的地理位置

### 3.3 GeoPolygon（地理多边形）

查询指定多边形区域内的点：

```json
{
  "key": "location",
  "geo_polygon": {
    "exterior": {
      "points": [
        {"lon": -122.5, "lat": 37.8},
        {"lon": -122.3, "lat": 37.8},
        {"lon": -122.3, "lat": 37.6},
        {"lon": -122.5, "lat": 37.6}
      ]
    },
    "interiors": []  // 可选：内部多边形（孔洞）
  }
}
```

**用途**：查询复杂多边形区域内的地理位置

---

## 四、时间范围算子

### 4.1 DatetimeRange（日期时间范围）

查询指定时间范围内的点：

```json
{
  "key": "created_at",
  "datetime_range": {
    "gt": "2024-01-01T00:00:00Z",   // 大于
    "gte": "2024-01-01T00:00:00Z",  // 大于等于
    "lt": "2024-12-31T23:59:59Z",   // 小于
    "lte": "2024-12-31T23:59:59Z"   // 小于等于
  }
}
```

**支持的格式**：
- ISO 8601 格式
- Unix 时间戳

**示例**：
```json
// 查询 2024 年的数据
{
  "key": "created_at",
  "datetime_range": {
    "gte": "2024-01-01T00:00:00Z",
    "lt": "2025-01-01T00:00:00Z"
  }
}
```

---

## 五、特殊条件算子

### 5.1 HasId（ID 存在）

检查点 ID 是否在指定集合中：

```json
{
  "has_id": [1, 2, 3, 4, 5]
}
```

**用途**：根据点 ID 过滤

**示例**：
```json
// 只查询指定的点
{
  "has_id": [
    {"num": 1},
    {"num": 2},
    {"uuid": "550e8400-e29b-41d4-a716-446655440000"}
  ]
}
```

### 5.2 HasVector（向量存在）

检查点是否分配了向量：

```json
{
  "has_vector": "vector_name"  // 向量名称
}
```

**用途**：检查点是否有指定的向量

### 5.3 IsEmpty（空值）

检查字段是否为空：

```json
{
  "is_empty": {
    "key": "description"
  }
}
```

### 5.4 IsNull（NULL 值）

检查字段是否为 NULL：

```json
{
  "is_null": {
    "key": "optional_field"
  }
}
```

### 5.5 Nested（嵌套条件）

对嵌套对象进行过滤：

```json
{
  "nested": {
    "key": "user",
    "filter": {
      "must": [
        {"key": "age", "range": {"gte": 18}},
        {"key": "status", "match": {"value": "active"}}
      ]
    }
  }
}
```

**用途**：查询嵌套 JSON 对象中的字段

---

## 六、数学表达式算子（Expression）

用于计算评分和自定义排序，支持以下数学运算：

### 6.1 基础运算

#### Constant（常量）
```json
{
  "constant": 3.14
}
```

#### Variable（变量）
```json
{
  "variable": "price"  // 引用 payload 字段
}
```

#### Sum（求和）
```json
{
  "sum": [
    {"variable": "price"},
    {"variable": "shipping"},
    {"constant": 10}
  ]
}
```

#### Mult（乘法）
```json
{
  "mult": [
    {"variable": "price"},
    {"constant": 0.9}  // 打 9 折
  ]
}
```

#### Div（除法）
```json
{
  "div": {
    "left": {"variable": "total"},
    "right": {"variable": "count"},
    "by_zero_default": 0  // 除零时的默认值
  }
}
```

#### Neg（取反）
```json
{
  "neg": {"variable": "score"}
}
```

### 6.2 数学函数

#### Abs（绝对值）
```json
{
  "abs": {"variable": "difference"}
}
```

#### Sqrt（平方根）
```json
{
  "sqrt": {"variable": "value"}
}
```

#### Pow（幂运算）
```json
{
  "pow": {
    "base": {"variable": "x"},
    "exponent": {"constant": 2}  // x^2
  }
}
```

#### Exp（指数函数）
```json
{
  "exp": {"variable": "x"}  // e^x
}
```

#### Log10（常用对数）
```json
{
  "log10": {"variable": "value"}  // log10(x)
}
```

#### Ln（自然对数）
```json
{
  "ln": {"variable": "value"}  // ln(x)
}
```

### 6.3 衰减函数

#### ExpDecay（指数衰减）
```json
{
  "exp_decay": {
    "x": {"variable": "distance"},
    "target": {"constant": 0},
    "scale": 1000,      // 衰减尺度
    "midpoint": 0.5     // 中点值
  }
}
```

#### GaussDecay（高斯衰减）
```json
{
  "gauss_decay": {
    "x": {"variable": "distance"},
    "target": {"constant": 0},
    "scale": 1000,
    "midpoint": 0.5
  }
}
```

#### LinDecay（线性衰减）
```json
{
  "lin_decay": {
    "x": {"variable": "distance"},
    "target": {"constant": 0},
    "scale": 1000,
    "midpoint": 0.5
  }
}
```

### 6.4 特殊表达式

#### Condition（条件表达式）
```json
{
  "condition": {
    "key": "is_featured",
    "match": {"value": true}
  }
}
// 如果条件为真，返回 1.0；否则返回 0.0
```

#### GeoDistance（地理距离）
```json
{
  "geo_distance": {
    "origin": {"lon": -122.4, "lat": 37.7},
    "to": "location"  // payload 字段路径
  }
}
// 返回两点之间的距离（米）
```

#### Datetime（日期时间常量）
```json
{
  "datetime": "2024-01-01T00:00:00Z"
}
```

#### DatetimeKey（日期时间字段）
```json
{
  "datetime_key": "created_at"
}
```

---

## 七、组合使用示例

### 7.1 复杂过滤示例

```json
{
  "must": [
    {
      "key": "category",
      "match": {"keyword": "electronics"}
    },
    {
      "key": "price",
      "range": {"gte": 100, "lte": 1000}
    },
    {
      "key": "location",
      "geo_radius": {
        "center": {"lon": -122.4, "lat": 37.7},
        "radius": 5000
      }
    }
  ],
  "should": [
    {
      "key": "rating",
      "range": {"gte": 4.5}
    },
    {
      "key": "is_featured",
      "match": {"boolean": true}
    }
  ],
  "must_not": [
    {
      "key": "status",
      "match": {"keyword": "out_of_stock"}
    }
  ]
}
```

### 7.2 自定义评分示例

```json
{
  "score": {
    "sum": [
      {"variable": "vector_score"},
      {
        "mult": [
          {"variable": "rating"},
          {"constant": 0.3}
        ]
      },
      {
        "exp_decay": {
          "x": {
            "geo_distance": {
              "origin": {"lon": -122.4, "lat": 37.7},
              "to": "location"
            }
          },
          "scale": 10000,
          "midpoint": 0.5
        }
      }
    ]
  }
}
```

---

## 八、算子优先级和性能

### 8.1 索引优化

Qdrant 会自动为以下字段类型创建索引以优化查询：

- **关键字匹配**：关键字索引
- **全文搜索**：全文索引
- **数值范围**：范围索引
- **地理位置**：地理索引
- **日期时间**：时间索引

### 8.2 查询规划

Qdrant 的查询规划器会：
1. 分析过滤条件
2. 选择最优的索引
3. 优化查询执行顺序
4. 最小化扫描的数据量

### 8.3 性能建议

1. **使用索引字段**：为常用过滤字段创建索引
2. **避免全表扫描**：使用 `must` 条件限制搜索范围
3. **合理使用 should**：避免过多的 `should` 条件
4. **地理查询优化**：使用地理索引加速地理位置查询

---

## 九、总结

Qdrant 支持的算子类型：

| 类别 | 算子数量 | 主要用途 |
|------|---------|---------|
| **逻辑组合** | 4 个 | must, should, must_not, min_should |
| **字段匹配** | 10+ 种 | keyword, integer, text, phrase 等 |
| **数值比较** | 4 个 | gt, gte, lt, lte |
| **地理空间** | 3 个 | GeoBoundingBox, GeoRadius, GeoPolygon |
| **时间范围** | 4 个 | gt, gte, lt, lte（时间） |
| **数学运算** | 13+ 个 | 加减乘除、幂、对数、衰减等 |
| **特殊条件** | 5 个 | HasId, HasVector, IsEmpty, IsNull, Nested |

**总计**：40+ 种算子，支持复杂的查询和评分需求。

---

## 参考资料

- [Qdrant 官方文档 - 过滤](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant 官方文档 - 评分](https://qdrant.tech/documentation/concepts/scoring/)
- [Qdrant API 参考](https://qdrant.github.io/qdrant/redoc/index.html)

---

**最后更新**：2025年1月
