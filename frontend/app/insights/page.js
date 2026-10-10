"use client";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";

export default function InsightsPage() {
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState("");
  const [length, setLength] = useState("medium");
  const [tone, setTone] = useState("neutral");
  const [focus, setFocus] = useState("");
  const [insight, setInsight] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listDocuments()
      .then((r) => setDocs(r.items.filter((d) => d.status === "completed")))
      .catch((e) => setError(e.message));
  }, []);

  // When another document is chosen, show its saved insights if it has any (404 = none yet).
  async function onPick(id) {
    setDocId(id);
    setInsight(null);
    setError("");
    if (!id) return;
    try {
      setInsight(await api.getInsights(id));
    } catch (e) {
      if (e.status !== 404) setError(e.message);
    }
  }

  async function onGenerate() {
    setLoading(true);
    setError("");
    try {
      const options = { length, tone };
      if (focus.trim()) options.focus = focus.trim();
      setInsight(await api.generateInsights(docId, options));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <h1>Document insights</h1>
      <div className="card">
        <div className="row">
          <select value={docId} onChange={(e) => onPick(e.target.value)} style={{ maxWidth: 360 }}>
            <option value="">Choose a processed document...</option>
            {docs.map((d) => <option key={d.id} value={d.id}>{d.original_filename}</option>)}
          </select>
          <select value={length} onChange={(e) => setLength(e.target.value)} style={{ maxWidth: 120 }}>
            <option value="short">short</option>
            <option value="medium">medium</option>
            <option value="long">long</option>
          </select>
          <select value={tone} onChange={(e) => setTone(e.target.value)} style={{ maxWidth: 140 }}>
            <option value="neutral">neutral</option>
            <option value="executive">executive</option>
            <option value="simple">simple</option>
          </select>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <input
            type="text"
            value={focus}
            onChange={(e) => setFocus(e.target.value)}
            placeholder="Optional focus, for example: exclusions"
            style={{ flex: 1 }}
          />
          <button disabled={!docId || loading} onClick={onGenerate}>
            {loading ? "Generating..." : insight ? "Regenerate" : "Generate"}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </div>

      {insight && (
        <div className="card">
          <div className="row spread">
            <h2>Summary</h2>
            <span className="muted">
              {insight.generated_by} · category: <b>{insight.category}</b>
            </span>
          </div>
          <p>{insight.summary}</p>
          <h2>Key points</h2>
          <ul>{insight.key_points.map((k, i) => <li key={i}>{k}</li>)}</ul>
          <div>{insight.tags.map((t) => <span key={t} className="tag">{t}</span>)}</div>
          {insight.note && <p className="muted">{insight.note}</p>}
        </div>
      )}
    </>
  );
}
