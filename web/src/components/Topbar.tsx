type Props = {
  title: string;
  subtitle: string;
};

export default function Topbar({ title, subtitle }: Props) {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <span className="eyebrow">Revenue Monitor</span>
        <h1>{title}</h1>
        <p className="subtitle">{subtitle}</p>
      </div>
      <div className="topbar-actions">
        <div className="status-card">
          <span className="status-label">环境</span>
          <strong>本地</strong>
        </div>
        <div className="status-card status-ok">
          <span className="status-label">状态</span>
          <strong>就绪</strong>
        </div>
      </div>
    </header>
  );
}
