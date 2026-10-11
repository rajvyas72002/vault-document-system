// One place that talks to the FastAPI backend. Every page imports from here.
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// FastAPI errors look like {"detail": "text"} or {"detail": [{"msg": "..."}]}
function errorText(body, status) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  return `Request failed (${status})`;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error(`Cannot reach the API at ${API_URL}. Is uvicorn running?`);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* empty body */
  }
  if (!res.ok) {
    const err = new Error(errorText(body, res.status));
    err.status = res.status;
    throw err;
  }
  return { status: res.status, data: body };
}

const json = (method, payload) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload ?? {}),
});

export const api = {
  listDocuments: async () => (await request("/documents?limit=100")).data,
  uploadDocument: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return (await request("/documents/upload", { method: "POST", body: form })).data;
  },
  // 200 = finished inline (document), 202 = queued (job). Callers check `status`.
  processDocument: async (id) => request(`/documents/${id}/process`, { method: "POST" }),
  getJob: async (id) => (await request(`/jobs/${id}`)).data,
  listChats: async () => (await request("/chats")).data,
  createChat: async (documentIds) =>
    (await request("/chats", json("POST", { document_ids: documentIds }))).data,
  chatHistory: async (id) => (await request(`/chats/${id}/history`)).data,
  sendMessage: async (id, question) =>
    (await request(`/chats/${id}/messages`, json("POST", { question }))).data,
  // thumbs up / down on one answer; sending again replaces it, clearRating takes it back
  rateAnswer: async (chatId, messageId, rating, comment) =>
    (await request(`/chats/${chatId}/messages/${messageId}/feedback`, json("PUT", { rating, comment: comment || null }))).data,
  clearRating: async (chatId, messageId) => {
    await request(`/chats/${chatId}/messages/${messageId}/feedback`, { method: "DELETE" });
  },
  // thumbs up / down on a document's current summary
  rateInsight: async (documentId, rating, comment) =>
    (await request(`/documents/${documentId}/insights/feedback`, json("PUT", { rating, comment: comment || null }))).data,
  clearInsightRating: async (documentId) => {
    await request(`/documents/${documentId}/insights/feedback`, { method: "DELETE" });
  },
  getInsights: async (id) => (await request(`/documents/${id}/insights`)).data,
  generateInsights: async (id, options) =>
    (await request(`/documents/${id}/insights`, json("POST", options))).data,
  metrics: async () => {
    const [documents, processing, system, feedback] = await Promise.all([
      request("/metrics/documents"),
      request("/metrics/processing"),
      request("/metrics/system"),
      request("/metrics/feedback"),
    ]);
    return { documents: documents.data, processing: processing.data, system: system.data, feedback: feedback.data };
  },
};

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
