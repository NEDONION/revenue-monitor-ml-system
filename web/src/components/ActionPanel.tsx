type Props = {
  onTrain: () => void;
  onEval: () => void;
  onInfer: () => void;
  onStop: () => void;
  hint: string;
  progress: number;
  model: string;
  onModelChange: (value: string) => void;
  canEvaluate: boolean;
  canInfer: boolean;
  children?: React.ReactNode;
};

export default function ActionPanel({
  onTrain,
  onEval,
  onInfer,
  onStop,
  hint,
  progress,
  model,
  onModelChange,
  canEvaluate,
  canInfer,
  children
}: Props) {
  return (
    <div className="card control-card">
      <div className="card-header">
        <div>
          <h2>任务控制</h2>
          <p className="subtitle muted">训练、评估与推理一键触发</p>
        </div>
        <span className="chip">离线流程</span>
      </div>
      <div className="model-select">
        <span>模型选择</span>
        <select value={model} onChange={(e) => onModelChange(e.target.value)}>
          <option value="tft">TFT（推荐）</option>
          <option value="tcn">TCN（基线）</option>
        </select>
      </div>
      <div className="actions">
        <button className="primary" onClick={onTrain}>开始训练</button>
        <button
          className="secondary"
          onClick={onEval}
          disabled={!canEvaluate}
          title={canEvaluate ? "" : "未检测到模型文件"}
        >
          开始评估
        </button>
        <button
          className="ghost"
          onClick={onInfer}
          disabled={!canInfer}
          title={canInfer ? "" : "未检测到模型文件"}
        >
          开始推理
        </button>
        <button className="danger" onClick={onStop}>停止任务</button>
      </div>
      <div className="progress-block">
        <div className="progress-meta">
          <span className="muted">进度</span>
          <strong>{Math.round(progress)}%</strong>
        </div>
        <div className="progress">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
        </div>
        <div className="hint">{hint}</div>
      </div>
      {children ? <div className="panel-stack">{children}</div> : null}
    </div>
  );
}
