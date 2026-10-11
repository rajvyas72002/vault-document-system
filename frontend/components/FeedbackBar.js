"use client";
import { useState } from "react";

// 👍 / 👎 with an optional comment after a 👎. Used under chat answers and under document summaries.
// value: "up" | "down" | null     onRate(rating, comment) / onClear(): talk to the API
// onChange(rating, comment): lets the page update what it shows. Clicking the active thumb again takes the rating back.
export default function FeedbackBar({ label, value, savedComment, onRate, onClear, onChange }) {
  const [busy, setBusy] = useState(false);
  const [asking, setAsking] = useState(false);
  const [comment, setComment] = useState(savedComment || "");
  const [err, setErr] = useState("");

  async function rate(rating, text) {
    setBusy(true);
    setErr("");
    try {
      if (!text && value === rating) {
        await onClear();
        onChange(null, null);
        setAsking(false);
      } else {
        await onRate(rating, text);
        onChange(rating, text || null);
        setAsking(rating === "down" && !text);
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="feedback">
      <span className="muted">{label}</span>
      <button className={`thumb ${value === "up" ? "on" : ""}`} disabled={busy} aria-label="Helpful"
        aria-pressed={value === "up"} title="Helpful" onClick={() => rate("up")}>👍</button>
      <button className={`thumb ${value === "down" ? "on bad" : ""}`} disabled={busy} aria-label="Not helpful"
        aria-pressed={value === "down"} title="Not helpful" onClick={() => rate("down")}>👎</button>
      {value && !asking && <span className="muted">Thanks for the feedback.</span>}
      {asking && (
        <form className="row" style={{ width: "100%", marginTop: 6 }}
          onSubmit={(e) => { e.preventDefault(); rate("down", comment.trim()); setAsking(false); }}>
          <input type="text" value={comment} maxLength={500} placeholder="What was wrong? (optional)"
            onChange={(e) => setComment(e.target.value)} />
          <button type="submit" className="secondary" disabled={busy || !comment.trim()}>Send</button>
        </form>
      )}
      {err && <span className="error">{err}</span>}
    </div>
  );
}
