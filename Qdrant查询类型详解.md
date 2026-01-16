# Qdrant 查询类型详解

## 概述

Qdrant 支持多种查询类型，从基础的最近邻搜索到复杂的融合查询。本文档详细梳理每一种查询类型，提供示例代码，并从源码和系统架构层面分析其实现逻辑。

---

## 一、查询类型总览

### 1.1 API 层面查询类型

**代码位置**: `lib/api/src/rest/schema.rs`

```627:656:lib/api/src/rest/schema.rs
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Query {
    /// Find the nearest neighbors to this vector.
    Nearest(NearestQuery),

    /// Use multiple positive and negative vectors to find the results.
    Recommend(RecommendQuery),

    /// Search for nearest points, but constrain the search space with context
    Discover(DiscoverQuery),

    /// Return points that live in positive areas.
    Context(ContextQuery),

    /// Order the points by a payload field.
    OrderBy(OrderByQuery),

    /// Fuse the results of multiple prefetches.
    Fusion(FusionQuery),

    /// Apply reciprocal rank fusion to multiple prefetches
    Rrf(RrfQuery),

    /// Score boosting via an arbitrary formula
    Formula(FormulaQuery),

    /// Sample points from the collection, non-deterministically.
    Sample(SampleQuery),
}
```

### 1.2 Segment 层面查询类型

**代码位置**: `lib/shard/src/query/query_enum.rs`

```10:17:lib/shard/src/query/query_enum.rs
pub enum QueryEnum {
    Nearest(NamedQuery<VectorInternal>),
    RecommendBestScore(NamedQuery<RecoQuery<VectorInternal>>),
    RecommendSumScores(NamedQuery<RecoQuery<VectorInternal>>),
    Discover(NamedQuery<DiscoveryQuery<VectorInternal>>),
    Context(NamedQuery<ContextQuery<VectorInternal>>),
    FeedbackNaive(NamedQuery<NaiveFeedbackQuery<VectorInternal>>),
}
```

### 1.3 查询类型列表

| 查询类型 | API 名称 | 说明 | Segment 层面 |
|---------|---------|------|-------------|
| **Nearest** | `Nearest` | 最近邻搜索 | `Nearest` |
| **Recommend** | `Recommend` | 推荐查询（基于正负样本） | `RecommendBestScore` / `RecommendSumScores` |
| **Discover** | `Discover` | 发现查询（带目标向量和上下文） | `Discover` |
| **Context** | `Context` | 上下文查询（仅上下文对） | `Context` |
| **OrderBy** | `OrderBy` | 按 Payload 字段排序 | - |
| **Prefetch** | `Prefetch` | 预取机制（支持嵌套） | - |
| **Fusion** | `Fusion` | 融合查询（合并多个 prefetch） | - |
| **Rrf** | `Rrf` | 参数化 Reciprocal Rank Fusion | - |
| **Formula** | `Formula` | 公式评分查询 | - |
| **Sample** | `Sample` | 随机采样查询 | - |
| **MMR** | `Nearest` + `mmr` | 最大边际相关性重排序 | - |

---

## 二、Nearest（最近邻搜索）

### 2.1 说明

**Nearest** 是 Qdrant 最基础的查询类型，用于查找与查询向量最相似的向量。

**特点**:
- 直接计算向量相似度
- 支持 HNSW 索引加速
- 支持量化搜索
- 支持 MMR 重排序

### 2.2 API 示例

```json
{
  "query": {
    "nearest": {
      "vector": [0.1, 0.2, 0.3, ...]
    }
  },
  "filter": {
    "must": [
      {"key": "category", "match": {"value": "electronics"}}
    ]
  },
  "limit": 10,
  "params": {
    "hnsw_ef": 128,
    "exact": false
  }
}
```

**Python 示例**:
```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

results = client.search(
    collection_name="products",
    query_vector=[0.1, 0.2, 0.3, ...],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}}
        ]
    },
    limit=10,
    search_params={
        "hnsw_ef": 128,
        "exact": False
    }
)
```

### 2.3 实现逻辑

#### 2.3.1 查询流程

```
用户请求
    ↓
Collection::core_search_batch()
    ↓
ShardHolder::select_shards()  // 选择目标分片
    ↓
ShardReplicaSet::core_search()  // 并行查询所有副本
    ↓
LocalShard::do_search()
    ↓
SegmentHolder::search()  // 在所有 Segment 中搜索
    ↓
Segment::search_batch()  // 每个 Segment 执行搜索
    ↓
VectorIndex::search()  // HNSW 或 Plain 索引搜索
    ↓
RawScorer::score_points()  // 计算相似度得分
    ↓
合并 Segment 结果
    ↓
合并 Shard 结果（K-way Merge）
    ↓
返回 Top-K 结果
```

#### 2.3.2 核心代码

**Segment 层搜索**:
```99:124:lib/segment/src/segment/search.rs
    #[cfg(feature = "testing")]
    pub fn search(
        &self,
        vector_name: &VectorName,
        vector: &QueryVector,
        with_payload: &WithPayload,
        with_vector: &WithVector,
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
    ) -> OperationResult<Vec<ScoredPoint>> {
        let query_context = QueryContext::default();
        let segment_query_context = query_context.get_segment_query_context();

        let result = self.search_batch(
            vector_name,
            &[vector],
            with_payload,
            with_vector,
            filter,
            top,
            params,
            &segment_query_context,
        )?;

        Ok(result.into_iter().next().unwrap())
    }
```

**向量评分**:
```274:281:lib/segment/src/vector_storage/raw_scorer.rs
        QueryVector::Nearest(vector) => {
            let query_scorer = MetricQueryScorer::<_, TMetric, _>::new(
                vector.try_into()?,
                vector_storage,
                hardware_counter_cell,
            );
            raw_scorer_from_query_scorer(query_scorer)
        }
```

#### 2.3.3 架构要点

1. **索引选择**: 根据 Segment 类型选择 HNSW 或 Plain 索引
2. **过滤处理**: 先过滤再搜索，或先搜索再过滤（取决于过滤选择性）
3. **量化支持**: 支持 PQ、SQ、BQ 量化加速搜索
4. **Oversampling**: 量化搜索时使用 oversampling 提高召回率
5. **结果合并**: 使用 K-way Merge 合并多 Segment/Shard 结果

---

## 三、Recommend（推荐查询）

### 3.1 说明

**Recommend** 查询使用多个正样本向量和负样本向量来查找结果。支持三种策略：
- **AverageVector**: 平均正负向量创建单个查询向量
- **BestScore**: 选择与候选点最相似的正样本或最不相似的负样本
- **SumScores**: 对所有正负样本的相似度求和（正样本加，负样本减）

### 3.2 API 示例

```json
{
  "query": {
    "recommend": {
      "positive": [1, 2, 3],  // 正样本点 ID
      "negative": [4, 5],     // 负样本点 ID
      "strategy": "best_score"
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.recommend(
    collection_name="products",
    positive=[1, 2, 3],  # 正样本点 ID
    negative=[4, 5],     # 负样本点 ID
    strategy="best_score",
    limit=10
)
```

### 3.3 实现逻辑

#### 3.3.1 RecommendBestScore 策略

**代码位置**: `lib/segment/src/vector_storage/query/reco_query.rs`

```67:90:lib/segment/src/vector_storage/query/reco_query.rs
impl<T> Query<T> for RecoBestScoreQuery<T> {
    fn score_by(&self, similarity: impl Fn(&T) -> ScoreType) -> ScoreType {
        // get similarities to all positives
        let positive_similarities = self.0.positives.iter().map(&similarity);

        // and all negatives
        let negative_similarities = self.0.negatives.iter().map(&similarity);

        // get max similarity to positives and max to negatives
        let max_positive = positive_similarities
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(ScoreType::NEG_INFINITY);

        let max_negative = negative_similarities
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(ScoreType::NEG_INFINITY);

        if max_positive > max_negative {
            scaled_fast_sigmoid(max_positive)
        } else {
            -scaled_fast_sigmoid(max_negative)
        }
    }
}
```

**评分逻辑**:
1. 计算候选点与所有正样本的最大相似度
2. 计算候选点与所有负样本的最大相似度
3. 如果 `max_positive > max_negative`，返回 `sigmoid(max_positive)`
4. 否则返回 `-sigmoid(max_negative)`

#### 3.3.2 RecommendSumScores 策略

```116:127:lib/segment/src/vector_storage/query/reco_query.rs
impl<T> Query<T> for RecoSumScoresQuery<T> {
    fn score_by(&self, similarity: impl Fn(&T) -> ScoreType) -> ScoreType {
        // Sum all positive vectors scores
        let positive_score: ScoreType = self.0.positives.iter().map(&similarity).sum();

        // Sum all negative vectors scores
        let negative_score: ScoreType = self.0.negatives.iter().map(&similarity).sum();

        // Subtract
        positive_score - negative_score
    }
}
```

**评分逻辑**:
1. 计算候选点与所有正样本的相似度之和
2. 计算候选点与所有负样本的相似度之和
3. 返回 `positive_score - negative_score`

#### 3.3.3 架构要点

1. **自定义评分器**: 使用 `CustomQueryScorer` 而非标准 `MetricQueryScorer`
2. **多向量比较**: 需要对每个候选点与所有正负样本进行比较
3. **HNSW 兼容**: 仍可使用 HNSW 索引，但评分逻辑不同

---

## 四、Discover（发现查询）

### 4.1 说明

**Discover** 查询用于在带上下文的约束空间中搜索最接近目标向量的点。

**特点**:
- 包含一个目标向量（target）
- 包含多个上下文对（context pairs）：每个对包含正样本和负样本
- 目标是找到既接近目标向量，又在正样本区域（远离负样本）的点

### 4.2 API 示例

```json
{
  "query": {
    "discover": {
      "target": [0.1, 0.2, 0.3, ...],
      "context": [
        {
          "positive": [1, 2],  // 应该接近的点
          "negative": [3, 4]   // 应该远离的点
        }
      ]
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.discover(
    collection_name="products",
    target=[0.1, 0.2, 0.3, ...],
    context=[
        {
            "positive": [1, 2],
            "negative": [3, 4]
        }
    ],
    limit=10
)
```

### 4.3 实现逻辑

#### 4.3.1 DiscoveryQuery 结构

**代码位置**: `lib/segment/src/vector_storage/query/discovery_query.rs`

```27:51:lib/segment/src/vector_storage/query/discovery_query.rs
#[derive(Debug, Clone, PartialEq, Serialize, Hash)]
pub struct DiscoveryQuery<T> {
    pub target: T,
    pub pairs: Vec<ContextPair<T>>,
}

impl<T> DiscoveryQuery<T> {
    pub fn new(target: T, pairs: Vec<ContextPair<T>>) -> Self {
        Self { target, pairs }
    }

    pub fn flat_iter(&self) -> impl Iterator<Item = &T> {
        let pairs_iter = self.pairs.iter().flat_map(|pair| pair.iter());

        iter::once(&self.target).chain(pairs_iter)
    }

    fn rank_by(&self, similarity: impl Fn(&T) -> ScoreType) -> RankType {
        self.pairs
            .iter()
            .map(|pair| pair.rank_by(&similarity))
            // get overall rank
            .sum()
    }
}
```

#### 4.3.2 评分逻辑

DiscoveryQuery 的评分结合了目标向量相似度和上下文约束：

1. **目标相似度**: 与目标向量的相似度
2. **上下文排名**: 在每个上下文对中，如果更接近正样本则排名 +1，否则 -1
3. **综合评分**: `target_similarity + context_rank`

#### 4.3.3 架构要点

1. **两阶段搜索**: 先使用上下文对缩小搜索空间，再在约束空间中寻找目标
2. **平滑损失函数**: 使用 sigmoid 函数使搜索更平滑
3. **多向量评分**: 需要计算与目标向量和所有上下文对的相似度

---

## 五、Context（上下文查询）

### 5.1 说明

**Context** 查询仅使用上下文对（正样本和负样本）来查找结果，不使用目标向量。

**特点**:
- 只包含上下文对（正负样本对）
- 目标是找到在正样本区域（远离负样本）的点
- 通过最小化损失函数来评分

### 5.2 API 示例

```json
{
  "query": {
    "context": {
      "context": [
        {
          "positive": [1, 2],
          "negative": [3, 4]
        }
      ]
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    query={
        "context": {
            "context": [
                {
                    "positive": [1, 2],
                    "negative": [3, 4]
                }
            ]
        }
    },
    limit=10
)
```

### 5.3 实现逻辑

#### 5.3.1 ContextQuery 结构

**代码位置**: `lib/segment/src/vector_storage/query/context_query.rs`

```85:121:lib/segment/src/vector_storage/query/context_query.rs
#[derive(Debug, Clone, PartialEq, Serialize, Hash)]
pub struct ContextQuery<T> {
    pub pairs: Vec<ContextPair<T>>,
}

impl<T> ContextQuery<T> {
    pub fn new(pairs: Vec<ContextPair<T>>) -> Self {
        Self { pairs }
    }

    pub fn flat_iter(&self) -> impl Iterator<Item = &T> {
        self.pairs.iter().flat_map(|pair| pair.iter())
    }
}

// ... 省略 TransformInto 实现 ...

impl<T> Query<T> for ContextQuery<T> {
    fn score_by(&self, similarity: impl Fn(&T) -> ScoreType) -> ScoreType {
        self.pairs
            .iter()
            .map(|pair| pair.loss_by(&similarity))
            .sum()
    }
}
```

#### 5.3.2 损失函数

```53:63:lib/segment/src/vector_storage/query/context_query.rs
    pub fn loss_by(&self, similarity: impl Fn(&T) -> ScoreType) -> ScoreType {
        const MARGIN: ScoreType = ScoreType::EPSILON;

        let positive = similarity(&self.positive);
        let negative = similarity(&self.negative);

        let difference = positive - negative - MARGIN;

        fast_sigmoid(ScoreType::min(difference, 0.0))
    }
```

**评分逻辑**:
1. 对每个上下文对计算损失：`loss = sigmoid(min(positive - negative - margin, 0))`
2. 对所有对的损失求和
3. 分数越低越好（在正样本区域且远离负样本）

#### 5.3.3 架构要点

1. **纯上下文搜索**: 不依赖目标向量，完全基于正负样本的约束
2. **损失最小化**: 通过最小化损失函数找到满足约束的点
3. **平滑搜索**: 使用 sigmoid 函数使搜索过程更平滑

---

## 六、OrderBy（按字段排序查询）

### 6.1 说明

**OrderBy** 查询按 Payload 字段的值排序，而不是按向量相似度排序。

**特点**:
- 需要 Payload 字段有数值索引（numeric index）
- 支持升序（asc）和降序（desc）
- 支持 `start_from` 参数用于分页

### 6.2 API 示例

```json
{
  "query": {
    "order_by": {
      "key": "price",
      "direction": "desc",
      "start_from": 1000.0
    }
  },
  "filter": {
    "must": [
      {"key": "category", "match": {"value": "electronics"}}
    ]
  },
  "limit": 20
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    query={
        "order_by": {
            "key": "price",
            "direction": "desc",
            "start_from": 1000.0
        }
    },
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}}
        ]
    },
    limit=20
)
```

### 6.3 实现逻辑

#### 6.3.1 OrderBy 结构

**代码位置**: `lib/segment/src/data_types/order_by.rs`

```70:121:lib/segment/src/data_types/order_by.rs
#[derive(Copy, Clone, Debug, PartialEq, Deserialize, Serialize, JsonSchema, Hash)]
#[serde(rename_all = "snake_case")]
pub struct OrderBy {
    /// Payload key to order by
    pub key: JsonPath,

    /// Direction of ordering: `asc` or `desc`. Default is ascending.
    pub direction: Option<Direction>,

    /// Which payload value to start scrolling from. Default is the lowest value for `asc` and the highest for `desc`
    pub start_from: Option<StartFrom>,
}
```

#### 6.3.2 执行流程

**代码位置**: `lib/segment/src/segment/order_by.rs`

```16:84:lib/segment/src/segment/order_by.rs
    pub fn filtered_read_by_index_ordered(
        &self,
        order_by: &OrderBy,
        limit: Option<usize>,
        condition: &Filter,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Vec<(OrderValue, PointIdType)>> {
        let payload_index = self.payload_index.borrow();
        let id_tracker = self.id_tracker.borrow();

        let numeric_index = payload_index
            .field_indexes
            .get(&order_by.key)
            .and_then(|indexes| indexes.iter().find_map(|index| index.as_numeric()))
            .ok_or_else(|| OperationError::MissingRangeIndexForOrderBy {
                key: order_by.key.to_string(),
            })?;

        let cardinality_estimation = payload_index.estimate_cardinality(condition, hw_counter);

        let start_from = order_by.start_from();

        let values_ids_iterator = payload_index
            .iter_filtered_points(
                condition,
                &*id_tracker,
                &cardinality_estimation,
                hw_counter,
                is_stopped,
            )
            .flat_map(|internal_id| {
                // Repeat a point for as many values as it has
                numeric_index
                    .get_ordering_values(internal_id)
                    // But only those which start from `start_from`
                    .filter(|value| match order_by.direction() {
                        Direction::Asc => value >= &start_from,
                        Direction::Desc => value <= &start_from,
                    })
                    .map(move |ordering_value| (ordering_value, internal_id))
            })
            .filter_map(|(value, internal_id)| {
                id_tracker
                    .external_id(internal_id)
                    .map(|external_id| (value, external_id))
            });

        let page = match order_by.direction() {
            Direction::Asc => {
                let mut page = match limit {
                    Some(limit) => peek_top_smallest_iterable(values_ids_iterator, limit),
                    None => values_ids_iterator.collect(),
                };
                page.sort_unstable_by(|(value_a, _), (value_b, _)| value_a.cmp(value_b));
                page
            }
            Direction::Desc => {
                let mut page = match limit {
                    Some(limit) => peek_top_largest_iterable(values_ids_iterator, limit),
                    None => values_ids_iterator.collect(),
                };
                page.sort_unstable_by(|(value_a, _), (value_b, _)| value_b.cmp(value_a));
                page
            }
        };

        Ok(page)
    }
```

#### 6.3.3 架构要点

1. **数值索引要求**: 需要为排序字段创建数值索引（numeric index）
2. **过滤优先**: 先应用过滤条件，再按字段值排序
3. **流式排序**: 使用 `peek_top_smallest_iterable` 或 `peek_top_largest_iterable` 进行流式排序
4. **分页支持**: 通过 `start_from` 参数支持分页

---

## 七、Prefetch（预取机制）

### 7.1 说明

**Prefetch** 是 Qdrant 中一个强大的查询组合机制，允许在执行主查询之前先执行多个子查询（prefetch），然后将这些子查询的结果作为主查询的输入。

**特点**:
- 支持嵌套 prefetch（prefetch 内部可以包含 prefetch）
- 每个 prefetch 可以有自己的过滤条件、limit、score_threshold
- Prefetch 结果可以在主查询中使用（如 Fusion、RRF）
- 支持跨 Collection 查询（通过 `lookup_from`）

### 7.2 Prefetch 结构

**代码位置**: `lib/api/src/rest/schema.rs`

```750:781:lib/api/src/rest/schema.rs
#[derive(Debug, Serialize, Deserialize, JsonSchema, Validate)]
pub struct Prefetch {
    /// Sub-requests to perform first. If present, the query will be performed on the results of the prefetches.
    #[validate(nested)]
    #[serde(default, with = "MaybeOneOrMany")]
    #[schemars(with = "MaybeOneOrMany<Prefetch>")]
    pub prefetch: Option<Vec<Prefetch>>,

    /// Query to perform. If missing without prefetches, returns points ordered by their IDs.
    #[validate(nested)]
    pub query: Option<QueryInterface>,

    /// Define which vector name to use for querying. If missing, the default vector is used.
    pub using: Option<VectorNameBuf>,

    /// Filter conditions - return only those points that satisfy the specified conditions.
    #[validate(nested)]
    pub filter: Option<Filter>,

    /// Search params for when there is no prefetch
    #[validate(nested)]
    pub params: Option<SearchParams>,

    /// Return points with scores better than this threshold.
    pub score_threshold: Option<ScoreType>,

    /// Max number of points to return. Default is 10.
    #[validate(range(min = 1))]
    pub limit: Option<usize>,

    /// The location to use for IDs lookup, if not specified - use the current collection and the 'using' vector
    /// Note: the other collection vectors should have the same vector size as the 'using' vector in the current collection
```

### 7.3 执行流程

**代码位置**: `lib/collection/src/shards/local_shard/query.rs`

Prefetch 的执行采用递归方式：

```199:280:lib/collection/src/shards/local_shard/query.rs
    fn recurse_prefetch<'a>(
        &'a self,
        merge_plan: MergePlan,
        prefetch_holder: &'a PrefetchResults,
        search_runtime_handle: &'a Handle,
        timeout: Duration,
        depth: usize,
        hw_counter_acc: HwMeasurementAcc,
    ) -> BoxFuture<'a, CollectionResult<Vec<Vec<ScoredPoint>>>> {
        async move {
            let MergePlan {
                sources: plan_sources,
                rescore_stages,
            } = merge_plan;

            let start_time = std::time::Instant::now();
            let max_len = plan_sources.len();
            let mut sources = Vec::with_capacity(max_len);

            // We need to preserve the order of the sources for some fusion strategies
            for source in plan_sources {
                match source {
                    Source::SearchesIdx(idx) => {
                        sources.push(prefetch_holder.get(FetchedSource::Search(idx))?)
                    }
                    Source::ScrollsIdx(idx) => {
                        sources.push(prefetch_holder.get(FetchedSource::Scroll(idx))?)
                    }
                    Source::Prefetch(prefetch) => {
                        let merged = self
                            .recurse_prefetch(
                                *prefetch,
                                prefetch_holder,
                                search_runtime_handle,
                                timeout,
                                depth + 1,
                                hw_counter_acc.clone(),
                            )
                            .await?
                            .into_iter();
                        sources.extend(merged);
                    }
                }
            }
```

**执行步骤**:
1. **预取阶段**: 并行执行所有 prefetch 查询
2. **结果收集**: 收集所有 prefetch 的结果
3. **递归处理**: 如果 prefetch 内部还有 prefetch，递归处理
4. **主查询执行**: 在主查询中使用 prefetch 结果（如 Fusion、RRF）
5. **结果合并**: 合并所有结果并应用最终过滤和排序

### 7.4 使用场景

1. **多向量融合**: 使用不同的查询向量，然后融合结果
2. **多策略组合**: 结合不同的查询策略（Nearest、Recommend 等）
3. **分阶段搜索**: 先进行粗搜索，再进行精细搜索
4. **跨 Collection 查询**: 从一个 Collection 获取结果，在另一个 Collection 中使用

### 7.5 API 示例

```json
{
  "prefetch": [
    {
      "query": {
        "nearest": {"vector": [0.1, 0.2, ...]}
      },
      "limit": 100
    },
    {
      "query": {
        "recommend": {
          "positive": [1, 2, 3]
        }
      },
      "limit": 100
    }
  ],
  "query": {
    "fusion": "rrf"
  },
  "limit": 10
}
```

---

## 八、Fusion（融合查询）

### 8.1 说明

**Fusion** 查询用于合并多个 prefetch 查询的结果。

**支持的融合算法**:
- **RRF** (Reciprocal Rank Fusion): 默认 RRF
- **DBSF** (Distribution-Based Score Fusion): 基于分布的分数融合

**注意**: Fusion 查询必须与 prefetch 一起使用。

### 7.2 API 示例

```json
{
  "prefetch": [
    {
      "query": {
        "nearest": {"vector": [0.1, 0.2, ...]}
      },
      "limit": 100
    },
    {
      "query": {
        "nearest": {"vector": [0.3, 0.4, ...]}
      },
      "limit": 100
    }
  ],
  "query": {
    "fusion": "rrf"
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    prefetch=[
        {
            "query": {"nearest": {"vector": [0.1, 0.2, ...]}},
            "limit": 100
        },
        {
            "query": {"nearest": {"vector": [0.3, 0.4, ...]}},
            "limit": 100
        }
    ],
    query={"fusion": "rrf"},
    limit=10
)
```

### 7.3 实现逻辑

#### 7.3.1 Fusion 枚举

**代码位置**: `lib/api/src/rest/schema.rs`

```519:532:lib/api/src/rest/schema.rs
/// Fusion algorithm allows to combine results of multiple prefetches.
/// Available fusion algorithms:
///
/// * `rrf` - Reciprocal Rank Fusion (with default parameters)
/// * `dbsf` - Distribution-Based Score Fusion
#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Fusion {
    Rrf,
    Dbsf,
}
```

#### 7.3.2 融合执行

**代码位置**: `lib/collection/src/collection/query.rs`

```361:375:lib/collection/src/collection/query.rs
        let result = match query.as_ref() {
            Some(ScoringQuery::Fusion(fusion)) => {
                // If the root query is a Fusion, the returned results correspond to each the prefetches.
                let mut fused = match fusion {
                    FusionInternal::RrfK(k) => rrf_scoring(intermediates, *k),
                    FusionInternal::Dbsf => score_fusion(intermediates, ScoreFusion::dbsf()),
                };
                if let Some(&score_threshold) = score_threshold.as_ref() {
                    fused = fused
                        .into_iter()
                        .take_while(|point| point.score >= score_threshold.0)
                        .collect();
                }
                fused
            }
```

#### 7.3.3 RRF 算法

RRF (Reciprocal Rank Fusion) 公式：
```
RRF_score = Σ(1 / (k + rank_i))
```

其中：
- `k` 是 RRF 参数（默认 60）
- `rank_i` 是点在第 i 个 prefetch 结果中的排名

#### 7.3.4 架构要点

1. **两阶段查询**: 先执行 prefetch 查询，再融合结果
2. **Collection 层融合**: 融合在 Collection 层执行（合并所有 Shard 结果后）
3. **去重**: 自动处理多个 prefetch 结果中的重复点

---

## 九、Rrf（参数化 Reciprocal Rank Fusion）

### 9.1 说明

**Rrf** 是 **Fusion** 的参数化版本，允许自定义 `k` 参数。

**区别**:
- `Fusion::Rrf`: 使用默认 k 值（60）
- `Rrf`: 可以自定义 k 值

### 9.2 API 示例

```json
{
  "prefetch": [
    {"query": {"nearest": {"vector": [0.1, ...]}}, "limit": 100},
    {"query": {"nearest": {"vector": [0.2, ...]}}, "limit": 100}
  ],
  "query": {
    "rrf": {
      "k": 60
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    prefetch=[
        {"query": {"nearest": {"vector": [0.1, ...]}}, "limit": 100},
        {"query": {"nearest": {"vector": [0.2, ...]}}, "limit": 100}
    ],
    query={"rrf": {"k": 60}},
    limit=10
)
```

### 9.3 实现逻辑

**代码位置**: `lib/api/src/rest/schema.rs`

```533:537:lib/api/src/rest/schema.rs
/// Parameters for Reciprocal Rank Fusion
#[derive(Debug, Serialize, Deserialize, JsonSchema, Validate)]
pub struct Rrf {
    /// K parameter for reciprocal rank fusion
    #[validate(range(min = 1))]
    pub k: Option<u32>,
}
```

**执行逻辑**: 与 Fusion::Rrf 相同，但使用自定义 k 值。

---

## 十、Formula（公式评分查询）

### 10.1 说明

**Formula** 查询使用数学表达式对结果进行评分。

**支持的表达式**:
- 常量、变量（Payload 字段）
- 数学运算：加减乘除、幂、对数、指数、绝对值、平方根
- 条件表达式：如果条件为真则 1.0，否则 0.0
- 地理距离：两点之间的地理距离
- 衰减函数：基于距离或时间的衰减

### 9.2 API 示例

```json
{
  "query": {
    "formula": {
      "formula": {
        "sum": [
          {"variable": "score"},
          {"mult": [
            {"constant": 0.5},
            {"geo_distance": {
              "origin": {"lat": 52.5, "lon": 13.4},
              "to": "location"
            }}
          ]}
        ]
      },
      "defaults": {}
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    query={
        "formula": {
            "formula": {
                "sum": [
                    {"variable": "score"},
                    {"mult": [
                        {"constant": 0.5},
                        {"geo_distance": {
                            "origin": {"lat": 52.5, "lon": 13.4},
                            "to": "location"
                        }}
                    ]}
                ]
            }
        }
    },
    limit=10
)
```

### 10.3 实现逻辑

#### 10.3.1 Formula 结构

**代码位置**: `lib/shard/src/query/formula.rs`

```16:51:lib/shard/src/query/formula.rs
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FormulaInternal {
    pub formula: ExpressionInternal,
    pub defaults: HashMap<String, Value>,
}

impl TryFrom<FormulaInternal> for ParsedFormula {
    type Error = OperationError;

    fn try_from(value: FormulaInternal) -> Result<Self, Self::Error> {
        let FormulaInternal { formula, defaults } = value;

        let mut payload_vars = HashSet::new();
        let mut conditions = Vec::new();

        let parsed_expression = formula.parse_and_convert(&mut payload_vars, &mut conditions)?;

        let defaults = defaults
            .into_iter()
            .map(|(key, value)| {
                let key = key
                    .as_str()
                    .parse()
                    .map_err(|msg| failed_to_parse("variable ID", &key, &msg))?;
                OperationResult::Ok((key, value))
            })
            .try_collect()?;

        Ok(ParsedFormula {
            formula: parsed_expression,
            payload_vars,
            conditions,
            defaults,
        })
    }
}
```

#### 10.3.2 公式评分器

**代码位置**: `lib/segment/src/index/query_optimization/rescore_formula/formula_scorer.rs`

```97:110:lib/segment/src/index/query_optimization/rescore_formula/formula_scorer.rs
impl FormulaScorer<'_> {
    /// Evaluate the formula for the given point
    pub fn score(&self, point_id: PointOffsetType) -> OperationResult<ScoreType> {
        self.eval_expression(&self.formula, point_id)
            .and_then(|score| {
                let score_f32 = score as f32;
                if !score_f32.is_finite() {
                    return Err(OperationError::NonFiniteNumber {
                        expression: format!("{score} as f32 = {score_f32}"),
                    });
                }
                Ok(score_f32)
            })
    }
```

#### 10.3.3 架构要点

1. **表达式解析**: 将 JSON 表达式解析为内部表示（`ParsedFormula`）
2. **变量提取**: 从表达式中提取所有 Payload 变量
3. **条件提取**: 提取所有条件表达式用于预过滤优化
4. **递归求值**: 递归计算表达式值
5. **Payload 提供**: 需要 Payload Index 提供变量值

---

## 十一、Sample（随机采样查询）

### 11.1 说明

**Sample** 查询从 Collection 中随机采样点，不进行向量相似度搜索。

**特点**:
- 随机选择点
- 支持过滤条件
- 不依赖向量索引

### 11.2 API 示例

```json
{
  "query": {
    "sample": "random"
  },
  "filter": {
    "must": [
      {"key": "category", "match": {"value": "electronics"}}
    ]
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.query_points(
    collection_name="products",
    query={"sample": "random"},
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}}
        ]
    },
    limit=10
)
```

### 11.3 实现逻辑

#### 11.3.1 Sample 执行

**代码位置**: `lib/segment/src/segment/sampling.rs`

```12:38:lib/segment/src/segment/sampling.rs
    pub(super) fn filtered_read_by_index_shuffled(
        &self,
        limit: usize,
        condition: &Filter,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> Vec<PointIdType> {
        let payload_index = self.payload_index.borrow();
        let id_tracker = self.id_tracker.borrow();

        let cardinality_estimation = payload_index.estimate_cardinality(condition, hw_counter);
        let ids_iterator = payload_index
            .iter_filtered_points(
                condition,
                &*id_tracker,
                &cardinality_estimation,
                hw_counter,
                is_stopped,
            )
            .filter_map(|internal_id| id_tracker.external_id(internal_id));

        let mut rng = rand::rng();
        let mut shuffled = ids_iterator.choose_multiple(&mut rng, limit);
        shuffled.shuffle(&mut rng);

        shuffled
    }
```

#### 11.3.2 架构要点

1. **基于过滤**: 先应用过滤条件，再随机采样
2. **流式采样**: 使用 `choose_multiple` 从迭代器中选择
3. **随机打乱**: 对选中的点进行随机打乱
4. **不依赖向量**: 不需要向量索引

---

## 十二、MMR（最大边际相关性）

### 12.1 说明

**MMR** (Maximal Marginal Relevance) 是一种重排序算法，用于在相关性和多样性之间取得平衡。

**特点**:
- 作为 `Nearest` 查询的选项（通过 `mmr` 参数）
- 参数 `diversity` 控制相关性和多样性的平衡（0-1）
  - 较高值：偏向多样性（结果之间更不同）
  - 较低值：偏向相关性（与查询更相似）

### 11.2 API 示例

```json
{
  "query": {
    "nearest": {
      "vector": [0.1, 0.2, 0.3, ...],
      "mmr": {
        "diversity": 0.5,
        "candidates_limit": 100
      }
    }
  },
  "limit": 10
}
```

**Python 示例**:
```python
results = client.search(
    collection_name="products",
    query_vector=[0.1, 0.2, 0.3, ...],
    limit=10,
    mmr_threshold=0.5  # diversity 参数
)
```

### 12.3 实现逻辑

#### 12.3.1 MMR 结构

**代码位置**: `lib/api/src/rest/schema.rs`

```728:748:lib/api/src/rest/schema.rs
/// Maximal Marginal Relevance (MMR) algorithm for re-ranking the points.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(rename_all = "snake_case")]
pub struct Mmr {
    /// Tunable parameter for the MMR algorithm.
    /// Determines the balance between diversity and relevance.
    ///
    /// A higher value favors diversity (dissimilarity to selected results),
    /// while a lower value favors relevance (similarity to the query vector).
    ///
    /// Must be in the range [0, 1].
    /// Default value is 0.5.
    #[validate(range(min = 0.0, max = 1.0))]
    pub diversity: Option<f32>,

    /// The maximum number of candidates to consider for re-ranking.
    ///
    /// If not specified, the `limit` value is used.
    #[validate(range(max = 16_384))] // artificial maximum, to avoid too expensive query.
    pub candidates_limit: Option<usize>,
}
```

#### 12.3.2 MMR 算法

**代码位置**: `lib/shard/src/query/mmr/mod.rs`

```221:259:lib/shard/src/query/mmr/mod.rs
    // Iteratively select remaining points using MMR
    while selected_indices.len() < limit && !remaining_indices.is_empty() {
        let best_candidate = remaining_indices
            .iter()
            .map(|&candidate_idx| {
                let relevance_score = query_similarities[candidate_idx];

                debug_assert!(
                    selected_indices
                        .iter()
                        .all(|&selected_idx| selected_idx != candidate_idx)
                );

                // Find maximum similarity to any already selected point
                let max_similarity_to_selected = selected_indices
                    .iter()
                    .map(|selected_idx| {
                        similarity_matrix.get_similarity(candidate_idx, *selected_idx)
                    })
                    .max_by_key(|&sim| OrderedFloat(sim))
                    .unwrap_or(0.0);

                // Calculate MMR score: λ * relevance - (1 - λ) * max_similarity_to_selected
                let mmr_score =
                    lambda * relevance_score - (1.0 - lambda) * max_similarity_to_selected;

                (candidate_idx, mmr_score)
            })
            .max_by_key(|(_candidate_idx, mmr_score)| OrderedFloat(*mmr_score));

        if let Some((selected_idx, _mmr_score)) = best_candidate {
            // Select the best candidate and remove from remaining
            remaining_indices.swap_remove(&selected_idx);
            selected_indices.push(selected_idx);
        } else {
            break;
        }
    }
```

**MMR 公式**:
```
MMR_score = λ * relevance - (1 - λ) * max_similarity_to_selected
```

其中：
- `λ` (lambda) = `diversity` 参数
- `relevance` = 候选点与查询向量的相似度
- `max_similarity_to_selected` = 候选点与已选择点的最大相似度

#### 12.3.3 架构要点

1. **两阶段处理**:
   - **Shard 层**: 先执行 Nearest 搜索获取候选点（`candidates_limit`）
   - **Collection 层**: 在合并后的结果上执行 MMR 重排序
2. **相似度矩阵**: 需要计算候选点之间的相似度矩阵
3. **迭代选择**: 迭代选择具有最高 MMR 分数的候选点

---

## 十三、查询执行架构

### 13.1 查询流程总览

```
用户请求（REST/gRPC）
    ↓
API Handler（lib/api/src/rest/ 或 lib/api/src/grpc/）
    ↓
Dispatcher（lib/storage/src/content_manager/dispatcher.rs）
    ↓
Collection（lib/collection/src/collection/mod.rs）
    ↓
ShardHolder（lib/collection/src/shards/shard_holder/mod.rs）
    ↓
ShardReplicaSet（lib/collection/src/shards/replica_set/mod.rs）
    ↓
LocalShard（lib/collection/src/shards/local_shard/mod.rs）
    ↓
SegmentHolder（lib/collection/src/collection_manager/holders/segment_holder.rs）
    ↓
Segment（lib/segment/src/segment/mod.rs）
    ↓
VectorIndex（HNSW 或 Plain）
    ↓
VectorStorage（向量存储）
```

### 13.2 查询类型转换

```
API Query (lib/api/src/rest/schema.rs)
    ↓
Collection Query (lib/collection/src/operations/universal_query/collection_query.rs)
    ↓
Shard Query (lib/collection/src/operations/universal_query/shard_query.rs)
    ↓
QueryEnum (lib/shard/src/query/query_enum.rs)
    ↓
QueryVector (lib/segment/src/data_types/vectors.rs)
    ↓
RawScorer (lib/segment/src/vector_storage/raw_scorer.rs)
```

### 13.3 核心执行方法

#### 13.3.1 Collection 层

**代码位置**: `lib/collection/src/collection/query.rs`

```305:334:lib/collection/src/collection/query.rs
            )
            .await?;

        let results_f = transposed_iter(all_shards_results)
            .zip(requests_batch.iter())
            .map(|(shards_results, request)| async {
                // shards_results shape: [num_shards, num_intermediate_results, num_points]
                // merged_intermediates shape: [num_intermediate_results, num_points]
                let merged_intermediates = self
                    .merge_intermediate_results_from_shards(request, shards_results)
                    .await?;

                let result = self
                    .intermediates_to_final_list(
                        merged_intermediates,
                        request,
                        timeout.map(|timeout| timeout.saturating_sub(instant.elapsed())),
                        hw_measurement_acc.clone(),
                    )
                    .await?;

                let filter_refs = request.filter_refs();
                self.post_process_if_slow_request(instant.elapsed(), filter_refs);

                Ok::<_, CollectionError>(result)
            });
        let results = future::try_join_all(results_f).await?;

        Ok(results)
    }
```

#### 13.3.2 Shard 层

**代码位置**: `lib/collection/src/shards/local_shard/query.rs`

```199:242:lib/collection/src/shards/local_shard/query.rs
    fn recurse_prefetch<'a>(
        &'a self,
        merge_plan: MergePlan,
        prefetch_holder: &'a PrefetchResults,
        search_runtime_handle: &'a Handle,
        timeout: Duration,
        depth: usize,
        hw_counter_acc: HwMeasurementAcc,
    ) -> BoxFuture<'a, CollectionResult<Vec<Vec<ScoredPoint>>>> {
        async move {
            let MergePlan {
                sources: plan_sources,
                rescore_stages,
            } = merge_plan;

            let start_time = std::time::Instant::now();
            let max_len = plan_sources.len();
            let mut sources = Vec::with_capacity(max_len);

            // We need to preserve the order of the sources for some fusion strategies
            for source in plan_sources {
                match source {
                    Source::SearchesIdx(idx) => {
                        sources.push(prefetch_holder.get(FetchedSource::Search(idx))?)
                    }
                    Source::ScrollsIdx(idx) => {
                        sources.push(prefetch_holder.get(FetchedSource::Scroll(idx))?)
                    }
                    Source::Prefetch(prefetch) => {
                        let merged = self
                            .recurse_prefetch(
                                *prefetch,
                                prefetch_holder,
                                search_runtime_handle,
                                timeout,
                                depth + 1,
                                hw_counter_acc.clone(),
                            )
                            .await?
                            .into_iter();
                        sources.extend(merged);
                    }
                }
            }
```

#### 13.3.3 Segment 层

**代码位置**: `lib/segment/src/index/vector_index_search_common.rs`

```47:87:lib/segment/src/index/vector_index_search_common.rs
pub fn postprocess_search_result(
    mut search_result: Vec<ScoredPointOffset>,
    point_deleted: &BitSlice,
    vector_storage: &VectorStorageEnum,
    quantized_vectors: Option<&QuantizedVectors>,
    vector: &QueryVector,
    params: Option<&SearchParams>,
    top: usize,
    hardware_counter: HardwareCounterCell,
) -> OperationResult<Vec<ScoredPointOffset>> {
    let quantization_enabled = is_quantized_search(quantized_vectors, params);

    let default_rescoring = quantized_vectors
        .as_ref()
        .map(|q| q.default_rescoring())
        .unwrap_or(false);
    let rescore = quantization_enabled
        && params
            .and_then(|p| p.quantization)
            .and_then(|q| q.rescore)
            .unwrap_or(default_rescoring);
    if rescore {
        let mut scorer = FilteredScorer::new(
            vector.to_owned(),
            vector_storage,
            None,
            None,
            point_deleted,
            hardware_counter,
        )?;

        search_result = scorer
            .score_points(&mut search_result.iter().map(|x| x.idx).collect_vec(), 0)
            .collect();
        search_result.sort_unstable();
        search_result.reverse();
    }
    search_result.truncate(top);
    Ok(search_result)
}
```

---

## 十四、查询类型对比总结

### 14.1 功能对比表

| 查询类型 | 向量搜索 | 过滤支持 | 排序方式 | 使用场景 |
|---------|---------|---------|---------|---------|
| **Nearest** | ✅ | ✅ | 向量相似度 | 基础相似度搜索 |
| **Recommend** | ✅ | ✅ | 自定义评分 | 推荐系统 |
| **Discover** | ✅ | ✅ | 目标+上下文 | 上下文感知搜索 |
| **Context** | ✅ | ✅ | 上下文损失 | 纯上下文搜索 |
| **OrderBy** | ❌ | ✅ | Payload 字段 | 按字段排序 |
| **Prefetch** | ✅ | ✅ | 依赖子查询 | 查询组合机制 |
| **Fusion** | ✅ | ✅ | 融合分数 | 多查询融合 |
| **Rrf** | ✅ | ✅ | RRF 分数 | 多查询融合（可配置） |
| **Formula** | ❌ | ✅ | 公式评分 | 自定义评分逻辑 |
| **Sample** | ❌ | ✅ | 随机 | 随机采样 |
| **MMR** | ✅ | ✅ | MMR 分数 | 多样性与相关性平衡 |

### 14.2 性能对比

| 查询类型 | 索引使用 | 计算复杂度 | 内存使用 |
|---------|---------|-----------|---------|
| **Nearest** | HNSW/Plain | O(log N) | 低 |
| **Recommend** | HNSW/Plain | O(M * log N) | 低 |
| **Discover** | HNSW/Plain | O(M * log N) | 低 |
| **Context** | HNSW/Plain | O(M * log N) | 低 |
| **OrderBy** | Payload Index | O(N log N) | 中 |
| **Prefetch** | 依赖子查询 | O(Σ(M_i * log N)) | 高 |
| **Fusion** | 依赖 prefetch | O(K * N) | 高 |
| **Rrf** | 依赖 prefetch | O(K * N) | 高 |
| **Formula** | Payload Index | O(N) | 中 |
| **Sample** | Payload Index | O(N) | 低 |
| **MMR** | HNSW/Plain | O(N²) | 高 |

其中：
- `N` = 候选点数量
- `M` = 正负样本数量
- `K` = prefetch 数量

---

## 十五、代码位置总结

### 15.1 API 层

| 组件 | 文件 | 说明 |
|------|------|------|
| **REST Query** | `lib/api/src/rest/schema.rs` | REST API 查询定义 |
| **gRPC Query** | `lib/api/src/grpc/proto/points.proto` | gRPC 查询定义 |

### 15.2 Collection 层

| 组件 | 文件 | 说明 |
|------|------|------|
| **Collection Query** | `lib/collection/src/operations/universal_query/collection_query.rs` | Collection 层查询转换 |
| **Shard Query** | `lib/collection/src/operations/universal_query/shard_query.rs` | Shard 层查询定义 |
| **Query Execution** | `lib/collection/src/collection/query.rs` | 查询执行逻辑 |

### 15.3 Shard 层

| 组件 | 文件 | 说明 |
|------|------|------|
| **QueryEnum** | `lib/shard/src/query/query_enum.rs` | Segment 层查询枚举 |
| **Query Execution** | `lib/collection/src/shards/local_shard/query.rs` | Shard 层查询执行 |

### 15.4 Segment 层

| 组件 | 文件 | 说明 |
|------|------|------|
| **QueryVector** | `lib/segment/src/data_types/vectors.rs` | 查询向量类型 |
| **RecoQuery** | `lib/segment/src/vector_storage/query/reco_query.rs` | 推荐查询实现 |
| **DiscoveryQuery** | `lib/segment/src/vector_storage/query/discovery_query.rs` | 发现查询实现 |
| **ContextQuery** | `lib/segment/src/vector_storage/query/context_query.rs` | 上下文查询实现 |
| **RawScorer** | `lib/segment/src/vector_storage/raw_scorer.rs` | 原始评分器 |
| **OrderBy** | `lib/segment/src/segment/order_by.rs` | 排序查询实现 |
| **Sample** | `lib/segment/src/segment/sampling.rs` | 采样查询实现 |

### 15.5 特殊查询

| 组件 | 文件 | 说明 |
|------|------|------|
| **MMR** | `lib/shard/src/query/mmr/mod.rs` | MMR 算法实现 |
| **Formula** | `lib/shard/src/query/formula.rs` | 公式查询定义 |
| **Formula Scorer** | `lib/segment/src/index/query_optimization/rescore_formula/formula_scorer.rs` | 公式评分器 |

---

## 十六、总结

### 16.1 核心查询类型

Qdrant 支持 **11 种主要查询类型**（包括 Prefetch 机制），覆盖了从基础相似度搜索到复杂融合查询的各种场景：

1. **Nearest**: 基础最近邻搜索
2. **Recommend**: 基于正负样本的推荐
3. **Discover**: 上下文感知的发现查询
4. **Context**: 纯上下文约束查询
5. **OrderBy**: 按 Payload 字段排序
6. **Prefetch**: 预取机制（支持嵌套查询组合）
7. **Fusion**: 多查询结果融合（RRF/DBSF）
8. **Rrf**: 参数化 RRF 融合
9. **Formula**: 数学表达式评分
10. **Sample**: 随机采样
11. **MMR**: 最大边际相关性重排序

### 16.2 架构特点

1. **分层执行**: API → Collection → Shard → Segment
2. **并行处理**: 多 Shard 和多 Segment 并行执行
3. **结果合并**: K-way Merge 合并结果
4. **索引优化**: HNSW 索引加速向量搜索
5. **量化支持**: PQ/SQ/BQ 量化提高性能

### 16.3 设计优势

1. **灵活性**: 支持多种查询模式，适应不同场景
2. **性能**: 索引加速和量化优化提供高性能
3. **扩展性**: 支持分布式部署和水平扩展
4. **可组合性**: Prefetch + Fusion 支持复杂查询组合

---

## 参考资料

- [Qdrant 协调节点实现解读](./Qdrant协调节点实现解读.md)
- [分布式环境下 HNSW 向量搜索工作原理](./分布式环境下HNSW向量搜索工作原理.md)
- [Qdrant 支持的算子详解](./Qdrant支持的算子详解.md)
- [Collection、Shard、Segment 关系详解](./Collection_Shard_Segment关系详解.md)
