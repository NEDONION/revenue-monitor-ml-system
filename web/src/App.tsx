import { useEffect, useRef, useState } from "react";
import Topbar from "./components/Topbar";
import ActionPanel from "./components/ActionPanel";
import ChartCard from "./components/ChartCard";
import LogPanel from "./components/LogPanel";
import FileList from "./components/FileList";

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

export default function App() {
  const [progress, setProgress] = useState(0);
  const [hint, setHint] = useState("等待操作");
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<MetricState>(initialMetrics);
  const [trainLine, setTrainLine] = useState<number[]>([]);
  const [evalLine, setEvalLine] = useState<number[]>([]);
  const [polling, setPolling] = useState(false);
  const [jobName, setJobName] = useState<string>("-");
  const cursorRef = useRef(0);
  const [inferParams, setInferParams] = useState({
    steps: "24",
    site: "",
    currency: "",
    feeType: "",
    seriesId: ""
  });

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
            if (payload.status === "completed" && payload.job === "evaluate") {
              fetch("/reports/tcn_eval.json")
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
          }
        }
      } catch {
        // ignore
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [polling]);

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
    <div className="content">
      <Topbar title={activeMeta.title} subtitle={activeMeta.subtitle} />
      <section className="grid dashboard">
          <ActionPanel
            onTrain={() => runAction("/api/train", "训练中...")}
            onEval={() => runAction("/api/evaluate", "评估中...")}
            onInfer={() =>
              runAction("/api/infer", "推理中...", {
                steps: Number(inferParams.steps || "24"),
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
          >
            <div className="card-header compact">
              <h2>推理参数</h2>
              <span className="tag">可选</span>
            </div>
            <div className="form-grid">
              <label>
                预测步数（未来预测的小时数）
                <select
                  value={inferParams.steps}
                  onChange={(e) => setInferParams((prev) => ({ ...prev, steps: e.target.value }))}
                >
                  <option value="6">6 小时</option>
                  <option value="12">12 小时</option>
                  <option value="24">24 小时</option>
                  <option value="48">48 小时</option>
                  <option value="72">72 小时</option>
                </select>
              </label>
              <label>
                站点
                <input
                  value={inferParams.site}
                  onChange={(e) => setInferParams((prev) => ({ ...prev, site: e.target.value }))}
                  placeholder="US/UK/DE"
                />
              </label>
              <label>
                币种
                <input
                  value={inferParams.currency}
                  onChange={(e) => setInferParams((prev) => ({ ...prev, currency: e.target.value }))}
                  placeholder="USD/EUR"
                />
              </label>
              <label>
                费用类型
                <input
                  value={inferParams.feeType}
                  onChange={(e) => setInferParams((prev) => ({ ...prev, feeType: e.target.value }))}
                  placeholder="final_value_fee"
                />
              </label>
              <label>
                序列ID
                <input
                  value={inferParams.seriesId}
                  onChange={(e) => setInferParams((prev) => ({ ...prev, seriesId: e.target.value }))}
                  placeholder="site|currency|fee_type|metric_name|granularity"
                />
              </label>
            </div>
            <div className="hint muted">推理会自动读取模型元数据，步数不得超过训练 horizon。</div>
          </ActionPanel>
          <div className="card area-info">
            <div className="card-header">
              <h2>运行概览</h2>
              <span className="tag">实时</span>
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
            <div className="hint muted">
              支持一键停止正在运行的训练/评估/推理任务。
            </div>
          </div>
          <ChartCard
            title="训练概览"
            tag="TCN"
            chartId="train-chart"
            points={trainLine}
            stats={[
              { label: "train_loss", value: metrics.train },
              { label: "val_mae", value: metrics.mae },
              { label: "val_rmse", value: metrics.rmse }
            ]}
          />
          <ChartCard
            title="评估结果"
            tag="测试集"
            chartId="eval-chart"
            points={evalLine}
            stats={[
              { label: "mape", value: metrics.mape },
              { label: "smape", value: metrics.smape },
              { label: "r2", value: metrics.r2 }
            ]}
          />
          <LogPanel logs={logs} />
          <FileList />
      </section>
    </div>
  );
}
