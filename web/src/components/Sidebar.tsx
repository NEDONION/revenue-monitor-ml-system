type ItemKey = "dashboard" | "training" | "evaluation" | "inference";

const items: { key: ItemKey; label: string }[] = [
  { key: "dashboard", label: "总览" },
  { key: "training", label: "训练" },
  { key: "evaluation", label: "评估" },
  { key: "inference", label: "推理" }
];

type Props = {
  active: ItemKey;
  onSelect: (key: ItemKey) => void;
};

export default function Sidebar({ active, onSelect }: Props) {
  return (
    <aside className="sidebar">
      <div className="logo">
        <div className="logo-mark">RM</div>
        <div>
          <div className="logo-title">Revenue Monitor</div>
          <div className="logo-sub">ML System</div>
        </div>
      </div>
      <nav className="nav">
        {items.map((item) => (
          <button
            key={item.key}
            className={`nav-item ${active === item.key ? "active" : ""}`}
            onClick={() => onSelect(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="badge">Local</span>
        <span className="badge badge-ok">在线</span>
      </div>
    </aside>
  );
}
