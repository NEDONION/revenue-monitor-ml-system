# Revenue Monitor ML System

工程化的收入异常检测项目，使用单表数据驱动训练与推理，支持无监督的预测区间异常检测。

## 目标
- 多维度（站点/币种/费用类型）异常检测
- 预测区间判断异常（无异常标签依赖）
- 可离线训练、在线推理、可监控可回放

## 项目结构
- `docs/` 方案与设计文档
- `scripts/` 数据构造与离线任务脚本
- `data/` 本地数据目录（不提交）
- `src/models/` 按模型归档的训练/评估/推理实现（如 `tft`、`tcn`）

## 环境与依赖（uv）
本项目使用 `uv` 管理依赖，依赖列表统一放在 `requirements.txt`。

### 安装
```bash
uv pip install -r requirements.txt
```

### 锁文件（推荐）
如需生成锁文件（建议在有网络的环境下执行）：
```bash
uv lock
```

## 数据集构造
准备单表数据（字段要求见 `docs/tech_execution.md` 的“数据表结构建议”与“训练数据构造方案”）。

常用参数：
- `--input` 输入文件（CSV/Parquet），默认 `data/revenue_ts_wide.csv`
- `--output-dir` 输出目录，默认 `data/processed`
- `--granularity` 粒度：`minute | hourly | daily`，默认 `hourly`
- `--sample-rows` 仅在输入文件不存在时生成样例数据的行数，默认 `1_000_000`
- `--split` 训练/验证/测试比例，例如 `0.7,0.15,0.15`
- `--write-splits/--no-write-splits` 是否写出分片文件

示例（用已有数据直接构造）：
```bash
python3 scripts/prepare_dataset.py --input data/revenue_ts_wide.csv --granularity hourly
```

示例（生成 1_000_000 行样例数据后构造）：
```bash
python3 scripts/prepare_dataset.py --input data/revenue_ts_sample.csv --sample-rows 1_000_000
```

数字写法提示：
- `1000` 等价 `1_000`
- `1000000` 等价 `1_000_000`

## 训练（基线分位数模型）
```bash
python3 scripts/train_model.py
```

## 训练（TCN 时间序列模型）
```bash
python3 scripts/train_tcn.py
```

## 训练（TFT 时间序列模型）
```bash
python3 scripts/train_tft.py
```

可选：开启 W&B 训练可视化（GPU 机器同样适用）
```bash
export WANDB_PROJECT=revenue-monitor-ml-system
export WANDB_ENTITY=your-entity
export WANDB_MODE=online
```
如果不想同步远端：
```bash
export WANDB_MODE=offline
```

## 评估（TFT 时间序列模型）
```bash
python3 scripts/evaluate_tft.py
```

## 推理（TFT 时间序列模型）
```bash
python3 scripts/run_tft_inference.py
```

## 启动控制台与 API
```bash
python3 scripts/serve_api.py
```
然后访问 `http://localhost:8088/`。

## 前端（React + Vite）
```bash
cd web
npm install
npm run dev
```

## GPU 服务器快速开始
以下示例以全新 GPU 服务器为例，包含下载项目、构造数据、配置环境变量与启动训练。

```bash
# 1) 拉取项目
git clone https://github.com/NEDONION/revenue-monitor-ml-system
cd revenue-monitor-ml-system

# 安装 uv（若未安装）
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# 2) 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3) 构造数据（示例：1_000_000 行）
python3 scripts/prepare_dataset.py \
  --input data/revenue_ts_sample.csv \
  --sample-rows 1_000_000 \
  --granularity hourly

# 4) 配置环境变量（.env 可选）
cat << 'EOF' > .env
WANDB_PROJECT=revenue-monitor-ml-system
WANDB_ENTITY=your-entity
WANDB_MODE=online
EOF

# 5) 启动训练（TFT）
python3 scripts/train_tft.py
```

可选：如果不想同步到 W&B 服务端
```bash
export WANDB_MODE=offline
```

## 最佳实践（建议）
- 使用 `data/` 存放原始与处理后的数据，避免提交到版本库
- 训练与推理配置参数固定在脚本或配置文件中，避免口口相传
- 模型输出按版本号归档，支持回滚与复现

## 文档
- `docs/design_v0.md` 初始需求与背景
- `docs/tech_execution.md` 技术执行方案与数据准备流程
