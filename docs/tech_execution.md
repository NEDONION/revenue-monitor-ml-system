# 技术执行文档：收入异常检测系统（调研与落地方案）

## 1. 背景与目标
- 目标：对多站点/多币种/多费用类型收入进行异常检测，降低误报、提高异常响应速度。
- 范围：历史 2 年数据，分钟/小时/天粒度；多维组合建模；支持在线推理 API。
- 交付物：数据准备流水线、离线训练脚本、模型版本管理、在线推理服务、监控与告警接口。

## 2. 业务与数据调研结论
### 2.1 数据特征
- 强周期性：日/周/月季节性明显，促销和节假日波动显著。
- 多维度耦合：Site/Currency/FeeType 组合数量大，稀疏与非平稳并存。
- 异常类型：突刺（spike）、突降（drop）、漂移（drift）、延迟（lag）、重复/漏计费。

### 2.2 数据质量挑战
- 缺失点、延迟、异常 spike、币种换算波动。
- 维度组合长尾：部分组合历史数据不足。

## 3. 技术选型调研
### 3.1 模型路线对比
- 统计模型（Prophet/ARIMA）：解释性强，训练快；但对多维组合扩展性差。
- 深度模型（LSTM/TCN/Transformer）：可捕捉非线性与复杂周期；对长序列和多维特征更适合。
- 重构模型（AutoEncoder）：适合无监督异常检测，适应突发异常。

### 3.2 推荐方案（分层混合）
- 主模型：TCN 或 Transformer（多维输入、多序列并行）
- 兜底模型：简单季节性基线（如 STL+阈值）
- 异常判断：预测区间 vs 实际值的偏离 + 重构误差双指标
- 训练模式：按粒度分层训练（minute/hourly/daily 各自训练）
- 更新策略：定期再训练 + 增量回放评估

## 4. 系统架构
### 4.1 离线训练
- 数据准备：ETL -> 特征工程 -> 数据切分（train/val/test）
- 训练流程：多维组合切片 -> 并行训练 -> 模型评估 -> 版本化存储
- 模型管理：MLflow 或本地版本目录结构

### 4.2 在线推理
- 服务形式：Python API（FastAPI）
- 输入：时间序列窗口 + 维度标签
- 输出：是否异常 + 置信度/偏离度 + 维度信息
- 缓存策略：对短时间窗口做滑动缓存降低延迟
 - 请求模式：支持批量（多 series_id）与单条（单 series_id）两种

## 4.3 数据表结构建议（单表驱动）
以下为推荐的“长表”结构，满足训练、异常注入、评估、回放与线上推理的统一输入要求。

### 最低可用字段（必须具备）
- ts TIMESTAMP：时间点（建议 UTC）
- granularity VARCHAR：minute/hourly/daily
- site VARCHAR：US/UK/DE 等
- currency VARCHAR：USD/EUR 等
- fee_type VARCHAR：listing_fee / final_value_fee / payment_fee
- metric_name VARCHAR：建议固定为 revenue（可扩展）
- metric_value NUMERIC：指标数值（训练与异常判断核心）
- series_id VARCHAR：稳定主键（建议 `site|currency|fee_type|metric_name|granularity`）

### 建议补充字段（提升稳定性与可运维性）
- is_complete BOOLEAN：是否完整写入，避免延迟误报
- ingest_ts TIMESTAMP：数据落库时间，便于延迟分析
- source VARCHAR：数据源标识，便于问题定位
- is_holiday BOOLEAN：节假日标记（如有）
- exchange_rate NUMERIC：汇率（如需统一币种）

## 5. 关键技术执行方案
### 5.1 特征工程
- 时间特征：hour/day-of-week/holiday
- 维度编码：Site/Currency/FeeType（Embedding 或 One-hot）
- 历史窗口特征：rolling mean/std、季节性残差

### 5.2 模型训练
- 输入序列长度：根据粒度 7~28 天窗口
- 输出：未来 1~N 步预测区间
- 损失函数：Quantile Loss 或 MAE/MSE + 正则
- 并行策略：多 GPU/多进程 + 数据分片
 - 早停策略：验证集连续 5 轮无提升则停止
 - 版本命名：`model_{granularity}_{date}_{version}`

### 5.3 异常判定
- 规则：超出预测区间（p90/p10）或重构误差超阈值
- 多维聚合：对 Site/Currency 聚合级别提供二次验证
 - 置信度：偏离度归一化到 0~1（可视化使用）

### 5.4 监控与反馈
- 监控指标：模型误报率、漏报率、延迟
- 人工反馈闭环：异常确认后回流训练集

## 6. 训练数据构造方案（单表驱动）
目标：从 `revenue_ts_wide` 长表生成可直接用于预测区间模型的训练样本，兼顾多维分组、缺失处理与异常注入（可选）。

### 6.1 输入与筛选
- 输入表：`revenue_ts_wide`
- 时间范围：近 24 个月
- 粒度：按 `granularity` 分层处理（minute/hourly/daily 分开建模）
- 维度分组：`site`, `currency`, `fee_type`, `metric_name`
- 质量过滤：`is_complete = true`（若有）

### 6.2 基础清洗
- 去重：同一 `series_id + ts` 只保留最新 `ingest_ts`
- 缺失补齐：按粒度补齐时间轴；缺失点标记为 `is_missing = 1`，数值可先置 0 或插值
- 异常 spike 预处理：对极端值做 winsorize 或用滑动中位数裁剪

### 6.3 特征构造
- 时间特征：`hour`, `day_of_week`, `day_of_month`, `week_of_year`, `is_weekend`, `is_holiday`
- 统计特征：`rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`（窗口 7/14/28）
- 滞后特征：`lag_1`, `lag_7`, `lag_14`, `lag_28`（按粒度调整）
- 维度编码：`site`, `currency`, `fee_type`, `metric_name` -> embedding 或 one-hot

### 6.4 训练样本组织
- 预测窗口：`input_len = 7~28`（按粒度与业务需求调整）
- 预测步长：`horizon = 1~N`（建议 1 或 7）
- 输出目标：预测区间（p10/p90 或 p05/p95）
- 样本结构：`[features over input_len] -> [future value(s)]`

### 6.5 长尾维度处理
- 低频组合：按 `site` 或 `fee_type` 进行聚合建模
- 共享模型：对长尾维度使用全局模型 + 维度 embedding
- 过滤阈值：历史长度不足 `min_history` 的组合不单独建模

### 6.6 可选：异常注入（仅 POC）
- 异常类型：spike/drop/level_shift
- 注入比例：1%~3%
- 用途：模型效果 sanity check，不用于真实评估

### 6.7 数据切分
- 训练集：最早 70%
- 验证集：中间 15%
- 测试集：最新 15%
- 时间切分，禁止随机打乱

## 7. 合成数据生成方案（SDV）
目标：在真实业务数据不可用或不可外泄时，使用 SDV 生成结构一致、统计分布相近的合成数据，用于算法验证与流程打通（不作为业务效果验收依据）。

### 7.1 适用范围与限制
- 适用：管道/模型/服务端到端联调、特征工程与训练流程验证。
- 限制：合成数据不代表业务真实分布，异常召回/误报率仅做 sanity check。

### 7.2 表结构与元数据定义
合成表沿用 `revenue_ts_wide` 长表结构，字段含义保持一致。SDV 需要元数据以区分类型并保留约束。
- 主键：`series_id + ts`（复合主键）
- 时间列：`ts`
- 类别列：`granularity`, `site`, `currency`, `fee_type`, `metric_name`
- 数值列：`metric_value`, `exchange_rate`（如有）
- 布尔列：`is_complete`, `is_holiday`（如有）

### 7.3 生成策略（单表模型）
- 模型选择：`GaussianCopulaSynthesizer`（快速稳健）或 `CTGANSynthesizer`（捕捉非线性分布）
- 约束策略：
  - `metric_value >= 0`
  - `granularity` 取值限定在 minute/hourly/daily
  - `exchange_rate > 0`（如有）
- 时间分布：保持 `ts` 的时间跨度与粒度比例一致，确保序列连续性

### 7.4 时间序列一致性处理
SDV 单表模型不具备时间序列依赖，需要后处理保证序列合理性。
- 按 `series_id` 对 `ts` 排序，补齐缺失时间点
- 对 `metric_value` 做滚动平滑或 AR(1) 后处理以增强序列连续性
- 根据 `granularity` 规范化窗口长度（分钟/小时/日分别处理）

### 7.5 异常注入（合成数据专用）
- 类型：spike（突刺）、drop（突降）、level shift（均值漂移）
- 注入比例：1%~3%
- 目的：为异常检测提供可控样本，用于阈值与评估方法验证

### 7.6 产出规范
- 输出表名：`revenue_ts_synthetic`
- 数据规模：与真实目标规模同级或 10% 抽样规模
- 存储格式：CSV/Parquet 均可，优先 Parquet

## 8. 端到端执行细化（可落地清单）
### 8.1 数据准备阶段
- 输入：`revenue_ts_wide`（或 `revenue_ts_synthetic`）
- 输出：`dataset_{granularity}.parquet`（包含特征与标签）
- 操作步骤：
  - 校验字段完整性与类型
  - 生成 `series_id`
  - 补齐时间轴与缺失标记
  - 计算基础特征与滞后特征
  - 时间切分为 train/val/test
 - 脚本入口：`scripts/prepare_dataset.py`（支持 CSV/Parquet 输入）

### 8.2 训练阶段
- 输入：`dataset_{granularity}.parquet`
- 输出：`models/{granularity}/{version}/`
- 训练参数建议：
  - `input_len`: minute=1440, hourly=168, daily=56
  - `horizon`: 1 或 7
  - `quantiles`: [0.1, 0.5, 0.9] 或 [0.05, 0.5, 0.95]
  - `batch_size`: 256~1024（按显存调整）
  - `lr`: 1e-3
- 评估指标：MAE/RMSE + 区间覆盖率（PICP）

### 8.3 评估与回放
- 输入：test 集合
- 输出：离线评估报告（误报率、漏报率、覆盖率、延迟）
- 回放策略：按时间顺序滚动预测，记录异常点与偏离度

### 8.4 服务化与接口
- 输入：`series_id`, `ts`, `history_window`（或直接传全量窗口）
- 输出：
  - `is_anomaly`: bool
  - `confidence`: float
  - `lower_bound`, `upper_bound`, `y_pred`
  - `dimensions`: site/currency/fee_type
- SLA：单条推理 < 200ms；批量推理按条平均 < 20ms

### 8.5 监控与报警
- 运行指标：QPS、延迟、错误率
- 业务指标：异常触发率、异常持续时间、覆盖率
- 告警策略：连续 N 个点超界触发；或累计偏离度超阈值触发

### 8.6 模型治理
- 版本管理：模型与特征配置绑定
- 回滚策略：保留最近 3 个稳定版本
- 再训练频率：每周/每月按业务波动调整

## 6. 里程碑计划
### 阶段 1：POC（2~4 周）
- 数据抽样 -> 建模试验 -> 评估指标确定
- 输出：POC 报告 + 初版模型

### 阶段 2：MVP（4~6 周）
- 全量数据训练 -> API 推理服务 -> 基础告警
- 输出：MVP 系统上线

### 阶段 3：优化（持续）
- 引入更高维特征
- 自动阈值/自适应区间
- 缓存与延迟优化

## 7. 风险与对策
- 数据稀疏：长尾维度采用聚合 + 基线模型。
- 分布漂移：定期再训练 + 模型监控。
- 误报：预测区间动态扩展 + 业务规则融合。

## 8. 测试与验收
- 历史回放测试：已知异常事件是否检测到
- 线上影子模式：不告警，仅记录效果
- 验收指标：误报率降低 X%，漏报率控制在 Y% 内
