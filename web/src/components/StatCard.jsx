export default function StatCard({ label, value, trend }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {trend ? <span className="stat-trend">{trend}</span> : null}
    </div>
  );
}
