"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, formatBytes } from "../lib/api";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function DocumentsPage() {
  const [docs, setDocs] = useState([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState({}); // document id -> true while working
  const fileInput = useRef(null);

  const load = useCallback(async () => {
    try {
      setDocs((await api.listDocuments()).items);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setNotice("");
    try {
      await api.uploadDocument(file);
      setNotice(`Uploaded ${file.name}. Click Process to extract and index it.`);
      await load();
    } catch (err) {
      setError(err.message);
    }
    fileInput.current.value = "";
  }

  // 200 = done already (inline mode). 202 = queued: poll the job until it finishes.
  async function onProcess(doc) {
    setBusy((b) => ({ ...b, [doc.id]: true }));
    setError("");
    setNotice("");
    try {
      const res = await api.processDocument(doc.id);
      if (res.status === 202) {
        const jobId = res.data.job_id;
        for (let i = 0; i < 120; i++) {
          const job = await api.getJob(jobId);
          if (job.status === "succeeded" || job.status === "failed") {
            if (job.status === "failed") setError(`Processing failed: ${job.error_message}`);
            else setNotice(`${doc.original_filename}: ${job.chunk_count} chunks in ${job.duration_ms} ms.`);
            break;
          }
          await sleep(1000);
        }
      } else {
        setNotice(`${doc.original_filename} processed.`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy((b) => ({ ...b, [doc.id]: false }));
      load();
    }
  }

  return (
    <>
      <h1>Documents</h1>
      <div className="card">
        <div className="row spread">
          <div>
            <h2>Upload a PDF or DOCX</h2>
            <div className="muted">Then press Process. Scanned (image-only) PDFs are rejected.</div>
          </div>
          <input ref={fileInput} type="file" accept=".pdf,.docx" onChange={onUpload} />
        </div>
        {notice && <div className="notice">{notice}</div>}
        {error && <div className="error">{error}</div>}
      </div>

      <div className="card">
        <div className="row spread">
          <h2>Library ({docs.length})</h2>
          <button className="secondary" onClick={load}>Refresh</button>
        </div>
        {docs.length === 0 ? (
          <div className="muted">No documents yet.</div>
        ) : (
          <table>
            <thead>
              <tr><th>Name</th><th>Size</th><th>Pages</th><th>Status</th><th></th></tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id}>
                  <td>
                    {d.original_filename}
                    {d.error_message && <div className="error">{d.error_message}</div>}
                  </td>
                  <td>{formatBytes(d.file_size_bytes)}</td>
                  <td>{d.page_count ?? "-"}</td>
                  <td><span className={`badge ${d.status}`}>{d.status}</span></td>
                  <td>
                    <button disabled={busy[d.id] || d.status === "processing"} onClick={() => onProcess(d)}>
                      {busy[d.id] ? "Working..." : d.status === "completed" ? "Re-process" : d.status === "failed" ? "Retry" : "Process"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
