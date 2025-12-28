type Stat = { label: string; value: string };

type Props = {
  title: string;
  tag: string;
  stats: Stat[];
  points: number[];
  chartId: string;
};

const defaultPoints = [0];

function ChartLine({ points, id }: { points: number[]; id: string }) {
  const data = points.length ? points : defaultPoints;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const step = 320 / Math.max(1, data.length - 1);
  const d = data
    .map((val, idx) => {
      const x = idx * step;
      const y = 120 - ((val - min) / range) * (120 - 16) - 8;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg viewBox="0 0 320 120" aria-label={id}>
      <defs>
        <linearGradient id={`${id}-grad`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#1677ff" />
          <stop offset="100%" stopColor="#69c0ff" />
        </linearGradient>
      </defs>
      <path d={d} fill="none" stroke={`url(#${id}-grad)`} strokeWidth="3" />
    </svg>
  );
}

export default function ChartCard({ title, tag, stats, points, chartId }: Props) {
  return (
    <div className="card">
      <div className="card-header">
        <h2>{title}</h2>
        <span className="chip">{tag}</span>
      </div>
      <div className="metrics">
        {stats.map((stat) => (
          <div key={stat.label}>
            <span>{stat.label}</span>
            <strong>{stat.value}</strong>
          </div>
        ))}
      </div>
      <div className="chart">
        <ChartLine points={points} id={chartId} />
      </div>
    </div>
  );
}
