# v3 Outline — 干净重建（冲主会版）

> 目的：抛弃 v1/v2 的混乱，建一个职责清晰、可信、可扩展的代码骨架，直接对应
> `研究计划_冲主会.md` 的三条贡献（受控对照 → 局部性度量 → 自适应方法）。
> 本文件只是蓝图，按 milestone 分阶段实现，不一次写完。

---

## 0. 设计原则（吸取 v1/v2 的教训）

1. **受控对照是一等公民**：文档级与段落级**共用同一编码器、同一配置、全量检索**，差异只在粒度。这是论文 §5.2 的核心，也是 v1/v2 没做干净的地方。
2. **数据可信**：只用真实数据；加载失败**硬报错**，绝不静默 fallback 合成数据；相关文档**保证进候选池**（v1 的 MRR≈0.05 就是这个 bug）。
3. **评测用标准库**：MRR/nDCG/Recall 一律走 `ir_measures` / `pytrec_eval`，不手写；**内置配对显著性检验**（现稿零显著性是主会一票否决项）。
4. **编码器可插拔**：sparse(TF-IDF/BM25) / dense(BGE,E5) / late-interaction(ColBERT) 统一接口，换编码器不改实验代码。
5. **配置驱动、可复现**：所有实验由 YAML 配置定义；固定 seed、统一日志、结果落盘成结构化 JSON。
6. **小文件、单一职责**：每个模块只干一件事，便于测试与扩展。
7. **v1/v2 原地保留**：不删不移，作为"旧（不公平）对比"的复现参考；v3 是唯一新主线，后续开发只在 v3 进行。

---

## 1. 目录结构

```
v3/
├── OUTLINE.md            # 本文件
├── README.md            # 快速上手 + 实验清单
├── requirements.txt
├── configs/             # 实验配置（YAML，配置驱动）
│   ├── controlled_tfidf.yaml     # 受控对照：TF-IDF→GRP→L2（论文编码器）
│   ├── controlled_dense.yaml     # 受控对照：BGE/E5
│   ├── encoder_matrix.yaml       # 编码器 × 粒度 × 数据集 网格（Contribution 1）
│   └── adaptive.yaml             # 自适应粒度（Contribution 3）
├── src/
│   ├── config.py        # 加载+校验 YAML，转 dataclass
│   ├── data.py          # 真实数据加载（本地 BEIR / MS MARCO），无静默 fallback，正例进池
│   ├── encoders.py      # 统一编码器接口 + 三类实现
│   ├── segmentation.py  # 分段策略（自然段 / 句窗 / token chunk）
│   ├── aggregation.py   # 5 种聚合（max/topk_mean/weighted_topk/softmax/threshold_mean）
│   ├── retrieval.py     # DocumentLevelRetriever / ParagraphLevelRetriever（共用编码器，全量）
│   ├── metrics.py       # ir_measures 包装 + 配对显著性检验 + bootstrap CI
│   ├── locality.py      # 证据局部性度量（Contribution 2）
│   ├── adaptive.py      # 自适应粒度：路由 / 软融合（Contribution 3）
│   └── experiment.py    # 实验编排：读 config → 跑 → 存结果
├── scripts/
│   ├── run_controlled.py      # 单个受控对照
│   ├── run_encoder_matrix.py  # 编码器×粒度×数据集 网格
│   └── run_adaptive.py        # 自适应方法 vs 固定粒度
├── tests/               # pytest，每个 src 模块一份
└── results/             # 结构化结果输出
```

---

## 2. 模块职责（关键输入/输出）

### `src/data.py`
- `load_beir(name, n_docs, n_queries, seed, split="test", data_dir=...)` → `{passages, queries, qrels}`，qrels=`{qid:{doc_id:rel}}`，真实非空，正例保证进池。
- `load_msmarco(...)` 同上（ir_datasets）。
- 失败 `raise`；`allow_synthetic` 仅供单测。
- **可直接移植 v2 已修好的 data.py。**

### `src/encoders.py`
- 统一接口 `Encoder.fit(texts) / encode(texts)->np.ndarray(L2-normalized)`。
- `TfidfProjectionEncoder`（论文 Eq.1-3，纯 sklearn，无 GPU）→ M0 用。
- `DenseEncoder`（sentence-transformers：BGE/E5）→ M1 用。
- `ColbertEncoder`（late-interaction，可选）→ M1 强参照。
- 工厂 `build_encoder(cfg)`。

### `src/retrieval.py`（核心，最贴论文）
- `DocumentLevelRetriever(encoder)`：整篇→1 向量，全量余弦排序。
- `ParagraphLevelRetriever(encoder, segmenter, aggregation)`：每段编码→聚合→文档分→排序。
- 两者**同一 encoder 配置、各自 fit 自己的单元**（粒度变量）。
- **可由 v2/granularity.py 演进而来。**

### `src/metrics.py`
- `evaluate(run, qrels, ks)` → 走 `ir_measures`（MRR/nDCG/Recall/P/Hit）。
- `paired_test(per_query_a, per_query_b)` → 配对 t 检验 / 随机化检验 p 值。
- `bootstrap_ci(per_query)` → 置信区间。

### `src/locality.py`（Contribution 2）
- `evidence_locality(doc, query, qrels, segmenter, encoder)` → 标量：相关证据在文档中的"集中度"（候选定义：段落-查询相似度分布的熵/基尼/峰度）。
- `correlate(locality_scores, per_query_delta)` → R²/相关系数，证明它**预测**粒度收益。

### `src/adaptive.py`（Contribution 3）
- `GranularityRouter`：输入文档/查询特征（含 locality）→ 输出用文档级/段落级，或融合权重 α。
- `fuse(doc_score, para_score, alpha)` → 最终分。
- 目标：在强 baseline 上**同时打赢**固定文档级与固定段落级。

### `src/experiment.py` + `scripts/`
- 读 config → 加载数据 → 建两类检索器 → 评测 → 显著性 → 落盘 `results/<name>.json`（含每查询 delta，供 locality 分析）。

---

## 3. 实现 milestone（对应研究计划阶段）

| Milestone | 内容 | 产出 | 依赖 |
|---|---|---|---|
| **M0 地基** | config + data + encoders(TF-IDF) + retrieval + metrics(ir_measures+显著性) | 在 scifact/nfcorpus/fiqa 复现受控对照，带 p 值 | 无 GPU |
| **M1 编码器矩阵** | dense(BGE/E5)+ColBERT，多数据集网格 | Contribution 1：编码器×粒度收益表 | GPU |
| **M2 局部性度量** | locality.py + 相关性验证 | Contribution 2：可预测粒度收益的指标 | M1 结果 |
| **M3 自适应方法** | adaptive.py + 对比固定粒度 | Contribution 3：自适应打赢两个固定基线 | M2 |

> M0 用本地已有数据（scifact/nfcorpus/fiqa）+ TF-IDF，立即可跑、可验证新故事线（受控后段落级是否仍胜）。M1 起上 GPU。

---

## 4. 已定决策（2026-06-29）

1. **v1/v2 原地不动**：不删不移，仅新增 `v3/`。
2. **第一步范围 = M0 地基**：只做 config + data + TF-IDF 编码器 + 受控检索 + 评测，先在本地数据复现带显著性的对照，跑通再进 M1。
3. **评测库 = `ir_measures`**（BEIR 生态通用，与官方对齐）。
4. **配置格式 = YAML**。

---

## 5. M0 落地清单（下一步要建的文件）

```
v3/
├── requirements.txt          # numpy, scikit-learn, ir_measures, pyyaml, pytest
├── configs/
│   └── controlled_tfidf.yaml # 数据集列表 + TF-IDF→GRP(r=192) + 分段/聚合 + seed
├── src/
│   ├── __init__.py
│   ├── config.py             # YAML → dataclass，字段校验
│   ├── data.py               # 移植 v2 已修版（无静默 fallback、正例进池）
│   ├── encoders.py           # TfidfProjectionEncoder（统一接口，预留 dense/colbert）
│   ├── segmentation.py       # 自然段 / 句窗 / token chunk
│   ├── aggregation.py        # 5 种聚合
│   ├── retrieval.py          # DocumentLevel / ParagraphLevel（移植自 v2/granularity.py）
│   ├── metrics.py            # ir_measures 包装 + paired_test + bootstrap_ci
│   └── experiment.py         # 读 config → 跑两侧 → 显著性 → 存 results/
├── scripts/
│   └── run_controlled.py     # CLI 入口
├── tests/                    # 每模块一份 pytest
└── results/
```

**M0 验收标准**：`python scripts/run_controlled.py --config configs/controlled_tfidf.yaml`
在 scifact + nfcorpus + fiqa 上输出 document vs paragraph 的 MRR/nDCG/Recall + **配对检验 p 值**，
无 "synthetic data" 警告，结果落盘 `results/`。这一步直接检验"受控后段落级是否仍胜"这条新故事线。
