type Props = {
  logs: string[];
};

export default function LogPanel({ logs }: Props) {
  return (
    <div className="card log-card area-logs">
      <div className="card-header">
        <h2>任务日志</h2>
        <span className="tag">实时</span>
      </div>
      <div className="log">
        {logs.length === 0 ? (
          <div className="log-empty">暂无日志</div>
        ) : (
          logs.map((line, idx) => <div key={`${line}-${idx}`}>{line}</div>)
        )}
      </div>
    </div>
  );
}
