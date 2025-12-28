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

示例：
```bash
python3 scripts/prepare_dataset.py
```

## 训练（基线分位数模型）
```bash
python3 scripts/train_model.py
```

## 训练（TCN 时间序列模型）
```bash
python3 scripts/train_tcn.py
```

环境变量（可选，开启 W&B 训练可视化）：
```bash
cp .env.example .env
```

## 评估（TCN 时间序列模型）
```bash
python3 scripts/evaluate_tcn.py
```

## 推理（TCN 时间序列模型）
```bash
python3 scripts/run_inference.py
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

## 最佳实践（建议）
- 使用 `data/` 存放原始与处理后的数据，避免提交到版本库
- 训练与推理配置参数固定在脚本或配置文件中，避免口口相传
- 模型输出按版本号归档，支持回滚与复现

## 文档
- `docs/design_v0.md` 初始需求与背景
- `docs/tech_execution.md` 技术执行方案与数据准备流程
