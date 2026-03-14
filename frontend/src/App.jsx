import { useDeferredValue, useEffect, useState, startTransition } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const ANALYTICS_VIEWS = [
  { key: "phase", label: "Energy By Phase" },
  { key: "fleet", label: "Fleet Ranking" },
  { key: "importance", label: "Feature Importance" },
];

const PARAMETER_LABELS = {
  Temperature_C: "Temperature",
  Pressure_Bar: "Pressure",
  Humidity_Percent: "Humidity",
  Motor_Speed_RPM: "Motor Speed",
  Compression_Force_kN: "Compression Force",
  Flow_Rate_LPM: "Flow Rate",
  Vibration_mm_s: "Vibration",
};

function OptimalMarker({ cx, cy }) {
  const points = [
    [cx, cy - 12],
    [cx + 4, cy - 4],
    [cx + 12, cy - 3],
    [cx + 6, cy + 4],
    [cx + 8, cy + 12],
    [cx, cy + 7],
    [cx - 8, cy + 12],
    [cx - 6, cy + 4],
    [cx - 12, cy - 3],
    [cx - 4, cy - 4],
  ]
    .map(([x, y]) => `${x},${y}`)
    .join(" ");

  return (
    <g>
      <polygon points={points} fill="#ef4444" stroke="#ffffff" strokeWidth="2" />
    </g>
  );
}

function MetricCard({ label, value, hint }) {
  return (
    <article className="metric-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      {hint ? <p>{hint}</p> : null}
    </article>
  );
}

function SectionCard({ title, subtitle, children, actions }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function DataTable({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row[columns[0].key]}-${index}`}>
              {columns.map((column) => (
                <td key={column.key}>
                  {column.format ? column.format(row[column.key], row) : row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

async function fetchJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json();
}

export default function App() {
  const [bootstrap, setBootstrap] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const [selectedParameter, setSelectedParameter] = useState("Temperature_C");
  const [selectedAnalyticsView, setSelectedAnalyticsView] = useState("phase");
  const [loadingBootstrap, setLoadingBootstrap] = useState(true);
  const [loadingDashboard, setLoadingDashboard] = useState(false);
  const [loadingOptimization, setLoadingOptimization] = useState(false);
  const [error, setError] = useState("");
  const deferredBatchId = useDeferredValue(selectedBatchId);

  useEffect(() => {
    let cancelled = false;

    async function loadBootstrap() {
      setLoadingBootstrap(true);
      setError("");

      try {
        const payload = await fetchJson("/api/bootstrap");
        if (cancelled) {
          return;
        }

        setBootstrap(payload);
        setSelectedBatchId(payload.defaultBatchId ?? "");
        setSelectedParameter(payload.overview.parameterMeta[0]?.key ?? "Temperature_C");
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingBootstrap(false);
        }
      }
    }

    loadBootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!deferredBatchId) {
      return;
    }

    let cancelled = false;

    async function loadDashboard() {
      setLoadingDashboard(true);
      setOptimization(null);
      setError("");

      try {
        const payload = await fetchJson(`/api/batches/${deferredBatchId}`);
        if (!cancelled) {
          setDashboard(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingDashboard(false);
        }
      }
    }

    loadDashboard();

    return () => {
      cancelled = true;
    };
  }, [deferredBatchId]);

  useEffect(() => {
    if (!deferredBatchId) {
      return;
    }

    let cancelled = false;

    async function loadOptimization() {
      setLoadingOptimization(true);

      try {
        const payload = await fetchJson(`/api/batches/${deferredBatchId}/optimization`);
        if (!cancelled) {
          setOptimization(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingOptimization(false);
        }
      }
    }

    loadOptimization();

    return () => {
      cancelled = true;
    };
  }, [deferredBatchId]);

  async function handleOptimization() {
    if (!selectedBatchId) {
      return;
    }

    setLoadingOptimization(true);
    setError("");

    try {
      const payload = await fetchJson(`/api/batches/${selectedBatchId}/optimization`);
      setOptimization(payload);
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoadingOptimization(false);
    }
  }

  if (loadingBootstrap) {
    return <div className="app-shell status-view">Loading dashboard configuration...</div>;
  }

  if (!bootstrap) {
    return (
      <div className="app-shell status-view">
        <div>
          <div>Unable to initialize the dashboard.</div>
          {error ? <div className="error-banner">{error}</div> : null}
        </div>
      </div>
    );
  }

  const parameterMeta = bootstrap.overview.parameterMeta;
  const formatFeatureName = (feature) => PARAMETER_LABELS[feature] ?? feature.replaceAll("_", " ");
  const currentAnalyticsLabel =
    ANALYTICS_VIEWS.find((view) => view.key === selectedAnalyticsView)?.label ?? "Energy By Phase";

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Production Energy Monitoring</span>
          <h1>Manufacturing Optimization Control Room</h1>
          <p>
            Monitor batch energy behavior, review process conditions, and generate operational
            recommendations from a single plant-facing dashboard.
          </p>
        </div>
        <div className="hero-panel">
          <label htmlFor="batch-select">Active Batch</label>
          <select
            id="batch-select"
            value={selectedBatchId}
            onChange={(event) => {
              const nextBatchId = event.target.value;
              startTransition(() => {
                setSelectedBatchId(nextBatchId);
              });
            }}
          >
            {bootstrap.batchIds.map((batchId) => (
              <option key={batchId} value={batchId}>
                {batchId}
              </option>
            ))}
          </select>
          <p>Model source: {bootstrap.overview.modelSource}</p>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="metrics-grid">
        <MetricCard label="Total Batches" value={bootstrap.overview.totalBatches} />
        <MetricCard label="Total Records" value={bootstrap.overview.totalRecords} />
        <MetricCard
          label="Prediction MAE"
          value={`${numberFormatter.format(bootstrap.overview.predictionMae)} kW`}
        />
        <MetricCard
          label="Current Batch"
          value={dashboard?.batchId ?? selectedBatchId}
          hint={dashboard ? `Phase: ${dashboard.selectedPhase}` : "Loading"}
        />
      </section>

      {loadingDashboard || !dashboard ? (
        <div className="status-view">Loading batch analytics...</div>
      ) : (
        <>
          <section className="metrics-grid">
            <MetricCard
              label="Actual Energy"
              value={`${numberFormatter.format(dashboard.metrics.actualEnergy)} kW`}
              hint="Measured power draw for the latest point in the selected batch."
            />
            <MetricCard
              label="Predicted Energy"
              value={`${numberFormatter.format(dashboard.metrics.predictedEnergy)} kW`}
              hint="Model-estimated energy demand for the current operating conditions."
            />
            <MetricCard
              label="Prediction Error"
              value={`${numberFormatter.format(dashboard.metrics.predictionError)} kW`}
              hint="Gap between measured consumption and model expectation."
            />
            <MetricCard
              label="Current Snapshot"
              value={dashboard.selectedPhase.replaceAll("_", " ")}
              hint={`${parameterMeta[0].label}: ${numberFormatter.format(dashboard.currentSnapshot[parameterMeta[0].key])}`}
            />
          </section>

          <div className="content-grid">
            <SectionCard
              title="Energy Trend"
              subtitle={`Actual and predicted energy profile for batch ${dashboard.batchId} across the production timeline.`}
            >
              <div className="chart-frame">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={dashboard.trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="x" stroke="#9fb2c8" />
                    <YAxis stroke="#9fb2c8" />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="actualEnergy"
                      stroke="#ff8c42"
                      strokeWidth={3}
                      dot={false}
                      name="Actual Energy"
                    />
                    <Line
                      type="monotone"
                      dataKey="predictedEnergy"
                      stroke="#38bdf8"
                      strokeWidth={3}
                      dot={false}
                      name="Predicted Energy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            <SectionCard
              title="Parameter Trend"
              subtitle="Review the main process variable trace for the selected batch without changing the overall dashboard context."
              actions={
                <select
                  className="mini-select"
                  value={selectedParameter}
                  onChange={(event) => setSelectedParameter(event.target.value)}
                >
                  {parameterMeta.map((parameter) => (
                    <option key={parameter.key} value={parameter.key}>
                      {parameter.label}
                    </option>
                  ))}
                </select>
              }
            >
              <div className="chart-frame">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={dashboard.trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="x" stroke="#9fb2c8" />
                    <YAxis stroke="#9fb2c8" />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey={selectedParameter}
                      stroke="#8b5cf6"
                      strokeWidth={3}
                      dot={false}
                      name={selectedParameter}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            <SectionCard
              title="Operational Alerts"
              subtitle="Phase-specific warning conditions based on historical upper operating thresholds, with immediate next-step guidance for the operator."
            >
              {dashboard.alerts.length ? (
                <div className="alert-stack">
                  {dashboard.alerts.map((alert) => (
                    <div className="alert-item" key={alert.metric}>
                      <strong>{formatFeatureName(alert.metric)}</strong>
                      <p>{alert.message}</p>
                      <p className="alert-insight">Recommended action: {alert.insight}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  No active threshold breaches detected for the current production phase.
                </div>
              )}
            </SectionCard>

            <SectionCard
              title={currentAnalyticsLabel}
              subtitle="Secondary analytics are grouped under one selector so the main monitoring workspace stays cleaner."
              actions={
                <select
                  className="mini-select analytics-select"
                  value={selectedAnalyticsView}
                  onChange={(event) => setSelectedAnalyticsView(event.target.value)}
                >
                  {ANALYTICS_VIEWS.map((view) => (
                    <option key={view.key} value={view.key}>
                      {view.label}
                    </option>
                  ))}
                </select>
              }
            >
              {selectedAnalyticsView === "phase" ? (
                <div className="chart-frame">
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={dashboard.phaseSummary}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="phaseLabel" stroke="#9fb2c8" />
                      <YAxis stroke="#9fb2c8" />
                      <Tooltip />
                      <Bar dataKey="averageEnergy" fill="#22c55e" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : null}

              {selectedAnalyticsView === "fleet" ? (
                <DataTable
                  columns={[
                    { key: "batchId", label: "Batch" },
                    {
                      key: "avgEnergy",
                      label: "Avg Energy",
                      format: (value) => `${numberFormatter.format(value)} kW`,
                    },
                    {
                      key: "peakEnergy",
                      label: "Peak Energy",
                      format: (value) => `${numberFormatter.format(value)} kW`,
                    },
                    {
                      key: "avgVibration",
                      label: "Avg Vibration",
                      format: (value) => `${numberFormatter.format(value)} mm/s`,
                    },
                  ]}
                  rows={dashboard.batchSummary}
                />
              ) : null}

              {selectedAnalyticsView === "importance" ? (
                <div className="chart-frame">
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={dashboard.featureImportances} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis type="number" stroke="#9fb2c8" />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={180}
                        tickFormatter={formatFeatureName}
                        stroke="#9fb2c8"
                      />
                      <Tooltip formatter={(value) => numberFormatter.format(value)} />
                      <Bar dataKey="importance" fill="#f97316" radius={[0, 8, 8, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : null}
            </SectionCard>
          </div>

          <SectionCard
            title="Optimization Advisor"
            subtitle="Recommendations refresh automatically whenever the active batch changes, so operators always see the latest optimized operating point."
            actions={
              <button
                className="primary-button"
                type="button"
                onClick={handleOptimization}
                disabled={loadingOptimization}
              >
                {loadingOptimization ? "Refreshing..." : "Refresh Recommendation"}
              </button>
            }
          >
            {loadingOptimization && !optimization ? (
              <div className="empty-state">
                Recomputing optimized operating values for the selected batch.
              </div>
            ) : optimization ? (
              <div className="optimization-layout">
                <div className="metrics-grid compact">
                  <MetricCard
                    label="Recommended Energy"
                    value={`${numberFormatter.format(optimization.bestCandidate.predictedEnergy)} kW`}
                    hint="Predicted energy demand at the recommended settings."
                  />
                  <MetricCard
                    label="Estimated Yield Uplift"
                    value={`${numberFormatter.format(optimization.bestCandidate.estimatedYieldUplift)} %`}
                    hint="AI-estimated productivity uplift based on the recommended operating state."
                  />
                  <MetricCard
                    label="Quality Confidence"
                    value={`${numberFormatter.format(optimization.bestCandidate.qualityConfidence)} %`}
                    hint={`AI confidence that the recommendation remains inside a stable quality envelope across ${optimization.bestCandidate.parametersAdjusted} key parameter changes.`}
                  />
                </div>

                <div className="chart-frame">
                  <ResponsiveContainer width="100%" height={360}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis
                        type="number"
                        dataKey="changeScore"
                        name="Change Score"
                        stroke="#9fb2c8"
                      />
                      <YAxis
                        type="number"
                        dataKey="predictedEnergy"
                        name="Predicted Energy"
                        stroke="#9fb2c8"
                      />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter data={optimization.paretoPoints} fill="#38bdf8" />
                      <Scatter data={[optimization.optimalPoint]} shape={<OptimalMarker />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <DataTable
                  columns={[
                    { key: "label", label: "Parameter" },
                    { key: "unit", label: "Unit" },
                    {
                      key: "current",
                      label: "Current",
                      format: (value, row) => `${numberFormatter.format(value)} ${row.unit}`,
                    },
                    {
                      key: "recommended",
                      label: "Recommended",
                      format: (value, row) => `${numberFormatter.format(value)} ${row.unit}`,
                    },
                    {
                      key: "delta",
                      label: "Delta",
                      format: (value, row) =>
                        `${value > 0 ? "+" : ""}${numberFormatter.format(value)} ${row.unit}`,
                    },
                  ]}
                  rows={optimization.comparison}
                />
              </div>
            ) : (
              <div className="empty-state">
                Run the recommendation search to generate Pareto-optimal operating points for the
                selected batch.
              </div>
            )}
          </SectionCard>
        </>
      )}
    </div>
  );
}
