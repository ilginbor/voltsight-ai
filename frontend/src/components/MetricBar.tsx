interface MetricBarProps {
  label: string;
  value: number;
  compact?: boolean;
}

export function MetricBar({
  label,
  value,
  compact = false,
}: MetricBarProps) {
  const boundedValue = Math.min(
    100,
    Math.max(
      0,
      value,
    ),
  );

  return (
    <div
      className={
        compact
          ? "metric metric--compact"
          : "metric"
      }
    >
      <div className="metric__header">
        <span>{label}</span>
        <strong>{value.toFixed(2)}</strong>
      </div>

      <div
        className="metric__track"
        aria-hidden="true"
      >
        <div
          className="metric__fill"
          style={{
            width: `${boundedValue}%`,
          }}
        />
      </div>
    </div>
  );
}
