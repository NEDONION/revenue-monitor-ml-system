const files = [
  { name: "tcn_eval.csv", url: "/reports/tcn_eval.csv", type: "CSV", note: "评估明细" },
  { name: "tcn_eval.json", url: "/reports/tcn_eval.json", type: "JSON", note: "评估摘要" },
  { name: "tcn_model.pt", url: "/models/tcn/tcn_model.pt", type: "PT", note: "模型权重" }
];

export default function FileList() {
  return (
    <div className="card">
      <div className="card-header">
        <h2>输出文件</h2>
        <span className="tag">reports</span>
      </div>
      <ul className="file-list">
        {files.map((file) => (
          <li key={file.name}>
            <span className="file-icon">{file.type}</span>
            <a href={file.url}>{file.name}</a>
            <span className="muted">{file.note}</span>
          </li>
        ))}
      </ul>
      <div className="hint muted">文件可用时点击下载或预览。</div>
    </div>
  );
}
