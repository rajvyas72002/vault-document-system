"use client";
import { useEffect, useState } from "react";
import { api, formatBytes } from "../../lib/api";

function Stat({ label, value }) {
  return (
    <div className="stat">
      <b>{value ?? "-"}</b>
      <span>{label}</span>
    </div>
  );
}

const pct = (x) => (x == null ? "-" : `${Math.round(x * 100)}%`);
const ms = (x) => (x == null ? "-" : `${Math.round(x)} ms`);

export default function MetricsPage() {
  const [m, setM] = useState(null);
  const [error, setError] = useState("");

  const load = () => api.metrics().then((r) => { setM(r); setError(""); }).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  if (error) return <div className="error">{error}</div>;
  if (!m) return <div className="muted">Loading...</div>;
  const { documents: d, processing: p, system: s } = m;

  return (
    <>
      <div className="row spread"><h1>Metrics</h1><button className="secondary" onClick={load}>Refresh</button></div>

      <div className="card">
        <h2>Documents</h2>
        <div className="grid">
          <Stat label="documents" value={d.total_documents} />
          <Stat label="pages" value={d.total_pages} />
          <Stat label="chunks" value={d.total_chunks} />
          <Stat label="with embeddings" value={d.chunks_with_embeddings} />
          <Stat label="with insights" value={d.documents_with_insights} />
          <Stat label="total size" value={formatBytes(d.total_size_bytes)} />
          <Stat label="uploaded last 24h" value={d.uploaded_last_24h} />
          <Stat label="avg chunks / doc" value={d.avg_chunks_per_document} />
        </div>
        <p className="muted">
          By status: {Object.entries(d.by_status).map(([k, v]) => `${k} ${v}`).join(", ") || "-"} ·
          By type: {Object.entries(d.by_type).map(([k, v]) => `${k} ${v}`).join(", ") || "-"}
        </p>
      </div>

      <div className="card">
        <h2>Processing</h2>
        <div className="grid">
          <Stat label="jobs" value={p.total_jobs} />
          <Stat label="success rate" value={pct(p.success_rate)} />
          <Stat label="avg duration" value={ms(p.avg_duration_ms)} />
          <Stat label="p95 duration" value={ms(p.p95_duration_ms)} />
          <Stat label={`stuck > ${p.stuck_threshold_minutes} min`} value={p.stuck_documents} />
        </div>
        {p.top_failure_reasons.length > 0 && (
          <p className="muted">
            Top failures: {p.top_failure_reasons.map((f) => `${f.reason} (${f.count})`).join("; ")}
          </p>
        )}
        <table>
          <thead><tr><th>Job</th><th>Status</th><th>Chunks</th><th>Duration</th><th>Error</th></tr></thead>
          <tbody>
            {p.recent_jobs.map((j) => (
              <tr key={j.id}>
                <td className="muted">{j.id.slice(0, 8)}</td>
                <td><span className={`badge ${j.status}`}>{j.status}</span></td>
                <td>{j.chunk_count ?? "-"}</td>
                <td>{ms(j.duration_ms)}</td>
                <td>{j.error_message ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Answers (mode: {s.ai_mode})</h2>
        <div className="grid">
          <Stat label="chat sessions" value={s.chat_sessions} />
          <Stat label="messages" value={s.chat_messages} />
          <Stat label="answers" value={s.answers_total} />
          <Stat label="answered" value={s.answers_answered} />
          <Stat label="refused" value={s.answers_refused} />
          <Stat label="answer rate" value={pct(s.answer_rate)} />
          <Stat label="avg latency" value={ms(s.avg_answer_latency_ms)} />
          <Stat label="p95 latency" value={ms(s.p95_answer_latency_ms)} />
        </div>
        <p className="muted">
          By generator: {Object.entries(s.answers_by_generator).map(([k, v]) => `${k} ${v}`).join(", ") || "-"}
        </p>
      </div>
    </>
  );
}
