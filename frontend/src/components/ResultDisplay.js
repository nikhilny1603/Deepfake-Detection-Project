import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import './ResultDisplay.css';

ChartJS.register(
  CategoryScale, LinearScale,
  BarElement, ArcElement,
  LineElement, PointElement,
  Title, Tooltip, Legend
);

export default function ResultDisplay({ result, modality }) {
  if (!result) return null;

  const isReal = result.prediction === 'Real';
  const confidence = Math.round(result.confidence * 100);
  const realPct = Math.round((result.details?.real_probability || 0) * 100);
  const fakePct = Math.round((result.details?.fake_probability || 0) * 100);

  return (
    <div className="result-container">
      {/* Verdict banner */}
      <div className={`verdict-banner ${isReal ? 'real' : 'fake'}`}>
        <div className="verdict-icon">{isReal ? '✅' : '⚠️'}</div>
        <div className="verdict-body">
          <div className="verdict-label">Detection Result</div>
          <div className="verdict-value">{result.prediction}</div>
        </div>
        <div className="verdict-confidence">
          <div className="conf-number">{confidence}%</div>
          <div className="conf-label">confidence</div>
        </div>
      </div>

      {/* Probability bars */}
      <div className="card">
        <h3 className="chart-title">Probability Breakdown</h3>
        <div className="prob-bars">
          <ProbBar label="Real" value={realPct} color="var(--success)" />
          <ProbBar label="Fake" value={fakePct} color="var(--danger)" />
        </div>
      </div>

      {/* Doughnut chart */}
      <div className="card chart-card">
        <h3 className="chart-title">Confidence Distribution</h3>
        <div className="doughnut-wrapper">
          <Doughnut
            data={{
              labels: ['Real', 'Fake'],
              datasets: [{
                data: [realPct, fakePct],
                backgroundColor: ['rgba(16,185,129,0.8)', 'rgba(239,68,68,0.8)'],
                borderColor: ['#10b981', '#ef4444'],
                borderWidth: 2,
              }]
            }}
            options={{
              plugins: {
                legend: { labels: { color: '#e2e8f0', font: { size: 13 } } },
                tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed}%` } }
              },
              cutout: '65%',
            }}
          />
        </div>
      </div>

      {/* Frame-level chart (video only) */}
      {modality === 'video' && result.details?.frame_scores?.length > 0 && (
        <FrameChart scores={result.details.frame_scores} />
      )}

      {/* Details table */}
      <DetailsTable details={result.details} modality={modality} />
    </div>
  );
}

function ProbBar({ label, value, color }) {
  return (
    <div className="prob-row">
      <div className="prob-label">{label}</div>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{ width: `${value}%`, background: color }}
        />
      </div>
      <div className="prob-pct">{value}%</div>
    </div>
  );
}

function FrameChart({ scores }) {
  const labels = scores.map((_, i) => `F${i + 1}`);
  return (
    <div className="card chart-card">
      <h3 className="chart-title">Frame-Level Fake Probability</h3>
      <Line
        data={{
          labels,
          datasets: [{
            label: 'Fake Probability',
            data: scores.map(s => Math.round(s * 100)),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239,68,68,0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: '#ef4444',
          }]
        }}
        options={{
          responsive: true,
          plugins: {
            legend: { labels: { color: '#e2e8f0' } },
            tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y}%` } }
          },
          scales: {
            x: {
              ticks: { color: '#94a3b8', maxTicksLimit: 16 },
              grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
              min: 0, max: 100,
              ticks: { color: '#94a3b8', callback: (v) => `${v}%` },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          }
        }}
      />
      <div className="threshold-note">— 50% threshold separates Real from Fake</div>
    </div>
  );
}

function DetailsTable({ details, modality }) {
  if (!details) return null;

  const IGNORE = ['frame_scores', 'frame_indices'];
  const entries = Object.entries(details).filter(([k]) => !IGNORE.includes(k));

  if (entries.length === 0) return null;

  return (
    <div className="card">
      <h3 className="chart-title">Analysis Details</h3>
      <table className="details-table">
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td className="detail-key">{formatKey(key)}</td>
              <td className="detail-val">{formatValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatValue(value) {
  if (typeof value === 'object' && value !== null) {
    return (
      <ul className="nested-list">
        {Object.entries(value).map(([k, v]) => (
          <li key={k}><strong>{formatKey(k)}:</strong> {String(v)}</li>
        ))}
      </ul>
    );
  }
  if (typeof value === 'number') {
    return value % 1 !== 0 ? value.toFixed(4) : String(value);
  }
  return String(value);
}
