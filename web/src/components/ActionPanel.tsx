type Props = {
  onTrain: () => void;
  onEval: () => void;
  onInfer: () => void;
  onStop: () => void;
  hint: string;
  progress: number;
  children?: React.ReactNode;
};

export default function ActionPanel({
  onTrain,
  onEval,
  onInfer,
  onStop,
  hint,
  progress,
  children
}: Props) {
  return (
    <div className="card">
      <div className="card-header">
        <h2>任务控制</h2>
        <span className="tag">离线流程</span>
      </div>
      <div className="actions">
        <button className="primary" onClick={onTrain}>开始训练</button>
        <button className="secondary" onClick={onEval}>开始评估</button>
        <button className="ghost" onClick={onInfer}>开始推理</button>
        <button className="danger" onClick={onStop}>停止任务</button>
      </div>
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      <div className="hint">{hint}</div>
      {children ? <div className="panel-section">{children}</div> : null}
    </div>
  );
}
