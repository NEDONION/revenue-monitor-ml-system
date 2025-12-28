import { useCallback, useEffect, useRef, useState } from "react";
import Topbar from "./components/Topbar";
import ActionPanel from "./components/ActionPanel";
import LogPanel from "./components/LogPanel";

type MetricState = {
  train: string;
  mae: string;
  rmse: string;
  mape: string;
  smape: string;
  r2: string;
};

type MetricParseResult = {
  epoch: number;
  total: number;
  train: number;
  val: number | null;
  valMae: number | null;
  valRmse: number | null;
};

const initialMetrics: MetricState = {
  train: "-",
  mae: "-",
  rmse: "-",
  mape: "-",
  smape: "-",
  r2: "-"
};

function parseMetrics(line: string): MetricParseResult | null {
  const match = line.match(
    /Epoch\s+(\d+)\/(\d+)\s+-\s+train:\s+([0-9.]+)(?:\s+-\s+val:\s+([0-9.]+)\s+-\s+val_mae:\s+([0-9.]+)\s+-\s+val_rmse:\s+([0-9.]+))?/
  );
  if (!match) return null;
  return {
    epoch: Number(match[1]),
    total: Number(match[2]),
    train: Number(match[3]),
    val: match[4] ? Number(match[4]) : null,
    valMae: match[5] ? Number(match[5]) : null,
    valRmse: match[6] ? Number(match[6]) : null
  };
}

const pageMeta = {
  title: "总览",
  subtitle: "训练、评估与推理的一站式概览。"
};

function getEvalReportPath(model: string) {
  return model === "tcn" ? "/reports/tcn_eval.json" : "/reports/tft_eval.json";
}

function getInferReportPath(model: string) {
  return model === "tcn" ? "/reports/tcn_inference.json" : "/reports/tft_inference.json";
}

export default function App() {
  const [progress, setProgress] = useState(0);
  const [hint, setHint] = useState("等待操作");
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<MetricState>(initialMetrics);
  const [trainLine, setTrainLine] = useState<number[]>([]);
  const [evalLine, setEvalLine] = useState<number[]>([]);
  const [polling, setPolling] = useState(false);
  const [jobName, setJobName] = useState<string>("-");
  const [modelKey, setModelKey] = useState("tft");
  const [modelAvailable, setModelAvailable] = useState(false);
  const [checkingModel, setCheckingModel] = useState(false);
  const cursorRef = useRef(0);
  const [inferParams, setInferParams] = useState({
    targetStep: "1",
    site: "US",
    currency: "USD",
    feeType: "listing_fee",
    seriesId: ""
  });
  const [sampleRows, setSampleRows] = useState("100000");
  const [sampleUnit, setSampleUnit] = useState("100000");
  const [inferResult, setInferResult] = useState<Record<string, string> | null>(null);

  const refreshModelStatus = useCallback(async (model: string) => {
    const target =
      model === "tft" ? "/models/tft/tft.ckpt" : "/models/tcn/tcn_model.pt";
    setCheckingModel(true);
    try {
      const res = await fetch(target, { method: "HEAD" });
      setModelAvailable(res.ok);
    } catch {
      setModelAvailable(false);
    } finally {
      setCheckingModel(false);
    }
  }, []);

  useEffect(() => {
    if (!polling) return;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/logs?start=${cursorRef.current}`);
        if (res.ok) {
          const payload = await res.json();
          setJobName(payload.job || "-");
          if (payload.lines.length) {
            setLogs((prev) => [...prev, ...payload.lines]);
            payload.lines.forEach((line: string) => {
              const parsed = parseMetrics(line);
              if (parsed) {
                setMetrics((prev) => ({
                  ...prev,
                  train: parsed.train.toFixed(4),
                  mae: parsed.valMae !== null ? parsed.valMae.toFixed(4) : prev.mae,
                  rmse: parsed.valRmse !== null ? parsed.valRmse.toFixed(4) : prev.rmse
                }));
                setTrainLine((prev) => [...prev, parsed.train]);
                if (parsed.total > 0) {
                  setProgress((parsed.epoch / parsed.total) * 100);
                }
              }
            });
          }
          cursorRef.current = payload.next;
          if (payload.status && payload.status !== "running") {
            setPolling(false);
            setHint(payload.status === "completed" ? "任务完成" : "任务结束");
            if (payload.status === "completed" && payload.job === "train") {
              refreshModelStatus(modelKey);
            }
            if (payload.status === "completed" && payload.job === "evaluate") {
              fetch(getEvalReportPath(modelKey))
                .then((r) => (r.ok ? r.json() : null))
                .then((data) => {
                  if (!data) return;
                  setMetrics((prev) => ({
                    ...prev,
                    mape: Number(data.mape).toFixed(4),
                    smape: Number(data.smape).toFixed(4),
                    r2: Number(data.r2).toFixed(4)
                  }));
                  setEvalLine((prev) => (prev.length ? prev : [data.mape, data.smape, data.r2]));
                })
                .catch(() => {});
            }
            if (payload.status === "completed" && payload.job === "infer") {
              fetch(getInferReportPath(modelKey))
                .then((r) => (r.ok ? r.json() : null))
                .then((data) => {
                  if (!data) return;
                  const prediction = Number(data.prediction ?? 0);
                  const lower = Number.isFinite(data.lower_bound) ? Number(data.lower_bound) : prediction;
                  const upper = Number.isFinite(data.upper_bound) ? Number(data.upper_bound) : prediction;
                  const q10 = Number.isFinite(data.q10_bound) ? Number(data.q10_bound) : prediction;
                  const q90 = Number.isFinite(data.q90_bound) ? Number(data.q90_bound) : prediction;
                  setInferResult({
                    series_id: String(data.series_id ?? ""),
                    last_ts: String(data.last_ts ?? ""),
                    horizon_step: String(data.horizon_step ?? ""),
                    prediction: prediction.toFixed(4),
                    lower_bound: lower.toFixed(4),
                    upper_bound: upper.toFixed(4),
                    q10_bound: q10.toFixed(4),
                    q90_bound: q90.toFixed(4)
                  });
                })
                .catch(() => {});
            }
          }
        }
      } catch {
        // ignore
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [polling]);

  useEffect(() => {
    setInferResult(null);
    setMetrics(initialMetrics);
    setTrainLine([]);
    setEvalLine([]);
    refreshModelStatus(modelKey);
    fetch(getEvalReportPath(modelKey))
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) return;
        setMetrics((prev) => ({
          ...prev,
          mape: Number(data.mape).toFixed(4),
          smape: Number(data.smape).toFixed(4),
          r2: Number(data.r2).toFixed(4)
        }));
        setEvalLine((prev) => (prev.length ? prev : [data.mape, data.smape, data.r2]));
      })
      .catch(() => {});

    fetch(getInferReportPath(modelKey))
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) return;
        const prediction = Number(data.prediction ?? 0);
        const lower = Number.isFinite(data.lower_bound) ? Number(data.lower_bound) : prediction;
        const upper = Number.isFinite(data.upper_bound) ? Number(data.upper_bound) : prediction;
        const q10 = Number.isFinite(data.q10_bound) ? Number(data.q10_bound) : prediction;
        const q90 = Number.isFinite(data.q90_bound) ? Number(data.q90_bound) : prediction;
        setInferResult({
          series_id: String(data.series_id ?? ""),
          last_ts: String(data.last_ts ?? ""),
          horizon_step: String(data.horizon_step ?? ""),
          prediction: prediction.toFixed(4),
          lower_bound: lower.toFixed(4),
          upper_bound: upper.toFixed(4),
          q10_bound: q10.toFixed(4),
          q90_bound: q90.toFixed(4)
        });
      })
      .catch(() => {});
  }, [modelKey, refreshModelStatus]);

  const runAction = async (url: string, label: string, body?: Record<string, unknown>) => {
    setHint(label);
    setProgress(0);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined
      });
      if (!res.ok) throw new Error("request failed");
      if (url === "/api/infer") {
        const data = await res.json();
        const prediction = Number(data.prediction ?? 0);
        const lower = Number.isFinite(data.lower_bound) ? Number(data.lower_bound) : prediction;
        const upper = Number.isFinite(data.upper_bound) ? Number(data.upper_bound) : prediction;
        const q10 = Number.isFinite(data.q10_bound) ? Number(data.q10_bound) : prediction;
        const q90 = Number.isFinite(data.q90_bound) ? Number(data.q90_bound) : prediction;
        setInferResult({
          series_id: String(data.series_id ?? ""),
          last_ts: String(data.last_ts ?? ""),
          horizon_step: String(data.horizon_step ?? ""),
          prediction: prediction.toFixed(4),
          lower_bound: lower.toFixed(4),
          upper_bound: upper.toFixed(4),
          q10_bound: q10.toFixed(4),
          q90_bound: q90.toFixed(4)
        });
        setHint("推理完成");
        return;
      }
      if (url === "/api/stop") {
        setPolling(false);
        setHint("已停止");
        return;
      }
      cursorRef.current = 0;
      setLogs([]);
      setPolling(true);
    } catch {
      setHint("请求失败，请检查后端服务");
    }
  };

  const activeMeta = pageMeta;

  return (
    <div className="app-shell">
      <Topbar title={activeMeta.title} subtitle={activeMeta.subtitle} />
      <div className="app-grid">
        <div className="stack">
          <ActionPanel
            onTrain={() => runAction("/api/train", "训练中...", { model: modelKey })}
            onEval={() => runAction("/api/evaluate", "评估中...", { model: modelKey })}
            onInfer={() =>
              runAction("/api/infer", "推理中...", {
                model: modelKey,
                target_step: Number(inferParams.targetStep || "24"),
                single_result: true,
                filters: {
                  site: inferParams.site || null,
                  currency: inferParams.currency || null,
                  fee_type: inferParams.feeType || null,
                  series_id: inferParams.seriesId || null
                }
              })
            }
            onStop={() => runAction("/api/stop", "停止中...")}
            hint={hint}
            progress={progress}
            model={modelKey}
            onModelChange={setModelKey}
            canEvaluate={modelAvailable && !checkingModel}
            canInfer={modelAvailable && !checkingModel}
          >
            <div className="panel-block">
              <div className="block-header">
                <h3>推理参数</h3>
                <span className="chip">可选</span>
              </div>
              <div className="form-grid">
                <label>
                  预测时间点（未来第 N 小时）
                  <select
                    value={inferParams.targetStep}
                    onChange={(e) => setInferParams((prev) => ({ ...prev, targetStep: e.target.value }))}
                  >
                    <option value="1">1 小时</option>
                    <option value="6">6 小时</option>
                    <option value="12">12 小时</option>
                    <option value="24">24 小时</option>
                    <option value="48">48 小时</option>
                  </select>
                </label>
                <label>
                  站点
                  <select
                    value={inferParams.site}
                    onChange={(e) => setInferParams((prev) => ({ ...prev, site: e.target.value }))}
                  >
                    <option value="">全部站点</option>
                    <option value="US">US</option>
                    <option value="UK">UK</option>
                    <option value="DE">DE</option>
                  </select>
                </label>
                <label>
                  币种
                  <select
                    value={inferParams.currency}
                    onChange={(e) => setInferParams((prev) => ({ ...prev, currency: e.target.value }))}
                  >
                    <option value="">全部币种</option>
                    <option value="USD">USD</option>
                    <option value="EUR">EUR</option>
                    <option value="GBP">GBP</option>
                  </select>
                </label>
                <label>
                  费用类型
                  <select
                    value={inferParams.feeType}
                    onChange={(e) => setInferParams((prev) => ({ ...prev, feeType: e.target.value }))}
                  >
                    <option value="">全部费用类型</option>
                    <option value="listing_fee">listing_fee</option>
                    <option value="final_value_fee">final_value_fee</option>
                    <option value="payment_processing_fee">payment_processing_fee</option>
                  </select>
                </label>
                <label>
                  序列ID
                  <select
                    value={inferParams.seriesId}
                    onChange={(e) => setInferParams((prev) => ({ ...prev, seriesId: e.target.value }))}
                  >
                    <option value="">自动（按过滤条件）</option>
                  </select>
                </label>
              </div>
              <div className="hint muted">推理会自动读取模型元数据，步数不得超过训练 horizon。</div>
            </div>
            <div className="panel-block">
              <div className="block-header">
                <h3>数据生成</h3>
                <span className="chip">可选</span>
              </div>
              <div className="form-grid">
                <label>
                  样本条数
                  <div className="inline-input">
                    <input
                      value={sampleRows}
                      onChange={(e) => setSampleRows(e.target.value.replace(/[^\d]/g, ""))}
                      placeholder="100"
                    />
                    <span className="inline-preview">
                      {Number(sampleRows || "0").toLocaleString()}
                    </span>
                    <select
                      value={sampleUnit}
                      onChange={(e) => setSampleUnit(e.target.value)}
                    >
                      <option value="10000">万 (10,000)</option>
                      <option value="1000000">百万 (1,000,000)</option>
                      <option value="10000000">千万 (10,000,000)</option>
                    </select>
                  </div>
                </label>
                <label className="form-span">
                  执行
                  <button
                    className="secondary"
                    onClick={() =>
                      runAction("/api/prepare", "生成数据中...", {
                        sample_rows: Number(sampleRows || "0") * Number(sampleUnit)
                      })
                    }
                  >
                    生成训练数据
                  </button>
                </label>
              </div>
            </div>
            <div className="panel-block">
              <div className="block-header">
                <h3>推理结果</h3>
                <span className="chip">最新</span>
              </div>
              {!inferResult ? (
                <div className="hint muted">暂无推理结果，请执行推理任务。</div>
              ) : (
                <div className="result-grid">
                  <div>
                    <span>序列标识</span>
                    <strong>{inferResult.series_id}</strong>
                  </div>
                  <div>
                    <span>最近时间</span>
                    <strong>{inferResult.last_ts}</strong>
                  </div>
                  <div>
                    <span>预测步</span>
                    <strong>{inferResult.horizon_step}</strong>
                  </div>
                  <div>
                    <span>预测值</span>
                    <strong>{inferResult.prediction}</strong>
                  </div>
                  <div>
                    <span>下界</span>
                    <strong>{inferResult.lower_bound}</strong>
                  </div>
                  <div>
                    <span>上界</span>
                    <strong>{inferResult.upper_bound}</strong>
                  </div>
                  <div>
                    <span>分位下界 (P10)</span>
                    <strong>{inferResult.q10_bound}</strong>
                  </div>
                  <div>
                    <span>分位上界 (P90)</span>
                    <strong>{inferResult.q90_bound}</strong>
                  </div>
                </div>
              )}
            </div>
          </ActionPanel>
        </div>
        <div className="stack main-stack">
          <section className="card overview-card">
            <div className="card-header">
              <div>
                <h2>训练与运行概览</h2>
                <p className="subtitle muted">监控训练损失与评估表现</p>
              </div>
              <span className="chip">{modelKey.toUpperCase()}</span>
            </div>
            <div className="info-grid">
              <div>
                <span>任务状态</span>
                <strong>{polling ? "运行中" : "空闲"}</strong>
              </div>
              <div>
                <span>当前任务</span>
                <strong>{jobName}</strong>
              </div>
              <div>
                <span>训练窗口</span>
                <strong>168 + 24</strong>
              </div>
              <div>
                <span>输出目录</span>
                <strong>reports/</strong>
              </div>
            </div>
            <div className="metrics">
              <div>
                <span>train_loss</span>
                <strong>{metrics.train}</strong>
              </div>
              <div>
                <span>val_mae</span>
                <strong>{metrics.mae}</strong>
              </div>
              <div>
                <span>val_rmse</span>
                <strong>{metrics.rmse}</strong>
              </div>
            </div>
            <div className="chart">
              <svg viewBox="0 0 320 120" aria-label="train-chart">
                <defs>
                  <linearGradient id="train-chart-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#0f766e" />
                    <stop offset="100%" stopColor="#5eead4" />
                  </linearGradient>
                </defs>
                <path
                  d={trainLine.length ? trainLine.map((val, idx) => {
                    const max = Math.max(...trainLine, 1);
                    const min = Math.min(...trainLine, 0);
                    const range = max - min || 1;
                    const step = 320 / Math.max(1, trainLine.length - 1);
                    const x = idx * step;
                    const y = 120 - ((val - min) / range) * (120 - 16) - 8;
                    return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
                  }).join(" ") : "M0,100 L320,100"}
                  fill="none"
                  stroke="url(#train-chart-grad)"
                  strokeWidth="3"
                />
              </svg>
            </div>
            <div className="panel-block">
              <div className="block-header">
                <h3>评估结果</h3>
                <span className="chip">测试集</span>
              </div>
              <div className="metrics">
                <div>
                  <span>mape</span>
                  <strong>{metrics.mape}</strong>
                </div>
                <div>
                  <span>smape</span>
                  <strong>{metrics.smape}</strong>
                </div>
                <div>
                  <span>r2</span>
                  <strong>{metrics.r2}</strong>
                </div>
              </div>
              <div className="chart">
                <svg viewBox="0 0 320 120" aria-label="eval-chart">
                  <defs>
                    <linearGradient id="eval-chart-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#f97316" />
                      <stop offset="100%" stopColor="#fdba74" />
                    </linearGradient>
                  </defs>
                  <path
                    d={evalLine.length ? evalLine.map((val, idx) => {
                      const max = Math.max(...evalLine, 1);
                      const min = Math.min(...evalLine, 0);
                      const range = max - min || 1;
                      const step = 320 / Math.max(1, evalLine.length - 1);
                      const x = idx * step;
                      const y = 120 - ((val - min) / range) * (120 - 16) - 8;
                      return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
                    }).join(" ") : "M0,100 L320,100"}
                    fill="none"
                    stroke="url(#eval-chart-grad)"
                    strokeWidth="3"
                  />
                </svg>
              </div>
            </div>
            <div className="hint muted">
              支持一键停止正在运行的训练/评估/推理任务。
            </div>
          </section>
          <LogPanel logs={logs} />
        </div>
      </div>
    </div>
  );
}
