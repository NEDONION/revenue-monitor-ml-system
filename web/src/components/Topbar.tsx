type Props = {
  title: string;
  subtitle: string;
};

export default function Topbar({ title, subtitle }: Props) {
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="topbar-actions">
        <span className="pill">环境：本地</span>
        <span className="pill pill-ok">就绪</span>
      </div>
    </header>
  );
}
