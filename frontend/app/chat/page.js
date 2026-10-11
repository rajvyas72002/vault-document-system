"use client";
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import FeedbackBar from "../../components/FeedbackBar";

function Citations({ items }) {
  if (!items || items.length === 0) return null;
  return items.map((c, i) => (
    <div className="cite" key={i}>
      <div className="muted">
        {c.document_name}{c.page_number ? `, page ${c.page_number}` : ""}
      </div>
      “{c.quote}”
    </div>
  ));
}

function Message({ m, chatId, onRated }) {
  const refused = m.role === "assistant" && m.answered === false;
  return (
    <div className={`msg ${m.role} ${refused ? "refused" : ""}`}>
      <div className="muted">
        {m.role === "user" ? "You" : "Assistant"}
        {m.role === "assistant" && m.generated_by ? ` · ${m.generated_by}` : ""}
        {m.latency_ms != null ? ` · ${m.latency_ms} ms` : ""}
        {refused ? " · not found in documents" : ""}
      </div>
      <div>{m.content}</div>
      <Citations items={m.citations} />
      {m.role === "assistant" && m.id !== "pending" && (
        <FeedbackBar
          label="Was this answer helpful?"
          value={m.feedback}
          savedComment={m.feedback_comment}
          onRate={(rating, text) => api.rateAnswer(chatId, m.id, rating, text)}
          onClear={() => api.clearRating(chatId, m.id)}
          onChange={(rating, text) => onRated(m.id, rating, text)}
        />
      )}
    </div>
  );
}

export default function ChatPage() {
  const [docs, setDocs] = useState([]);
  const [selected, setSelected] = useState({}); // document id -> true
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef(null);

  async function refreshChats() {
    setChats(await api.listChats());
  }

  useEffect(() => {
    (async () => {
      try {
        const all = (await api.listDocuments()).items;
        setDocs(all.filter((d) => d.status === "completed"));
        await refreshChats();
      } catch (e) {
        setError(e.message);
      }
    })();
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // keep the thumbs in the page state so the buttons update at once, without reloading the history
  function onRated(messageId, rating, comment) {
    setMessages((ms) => ms.map((x) => (x.id === messageId ? { ...x, feedback: rating, feedback_comment: comment } : x)));
  }

  async function openChat(id) {
    setChatId(id);
    setError("");
    try {
      setMessages((await api.chatHistory(id)).messages);
    } catch (e) {
      setError(e.message);
    }
  }

  async function newChat() {
    setError("");
    try {
      const ids = Object.keys(selected).filter((k) => selected[k]);
      const chat = await api.createChat(ids.length ? ids : null);
      await refreshChats();
      setChatId(chat.id);
      setMessages([]);
    } catch (e) {
      setError(e.message);
    }
  }

  async function onSend(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || !chatId || sending) return;
    setSending(true);
    setError("");
    setQuestion("");
    // show the user's message at once; replace with the stored history when the answer arrives
    setMessages((ms) => [...ms, { id: "pending", role: "user", content: q }]);
    try {
      await api.sendMessage(chatId, q);
      setMessages((await api.chatHistory(chatId)).messages);
      refreshChats();
    } catch (err) {
      setError(err.message);
      setMessages((ms) => ms.filter((m) => m.id !== "pending"));
      setQuestion(q);
    } finally {
      setSending(false);
    }
  }

  return (
    <>
      <h1>Chat with your documents</h1>
      <div className="chat">
        <aside className="side">
          <div className="card">
            <h2>Scope</h2>
            {docs.length === 0 && <div className="muted">No processed documents yet.</div>}
            {docs.map((d) => (
              <label key={d.id} className="check">
                <input
                  type="checkbox"
                  checked={!!selected[d.id]}
                  onChange={(e) => setSelected((s) => ({ ...s, [d.id]: e.target.checked }))}
                />
                <span>{d.original_filename}</span>
              </label>
            ))}
            <p className="muted">None ticked = search all documents.</p>
            <button onClick={newChat}>New chat</button>
          </div>
          <div className="card">
            <h2>Chats</h2>
            {chats.length === 0 && <div className="muted">None yet.</div>}
            {chats.map((c) => (
              <button
                key={c.id}
                className={c.id === chatId ? "" : "secondary"}
                onClick={() => openChat(c.id)}
              >
                {c.title || "New chat"}
              </button>
            ))}
          </div>
        </aside>

        <section className="card">
          {!chatId ? (
            <div className="muted">Choose the documents, press “New chat”, then ask a question.</div>
          ) : (
            <>
              {messages.length === 0 && <div className="muted">Ask your first question.</div>}
              {messages.map((m) => <Message key={m.id} m={m} chatId={chatId} onRated={onRated} />)}
              <div ref={bottom} />
              {error && <div className="error">{error}</div>}
              <form onSubmit={onSend} className="row" style={{ marginTop: 12 }}>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask about your documents..."
                  maxLength={1000}
                />
                <button type="submit" disabled={sending || question.trim().length < 3}>
                  {sending ? "Thinking..." : "Ask"}
                </button>
              </form>
            </>
          )}
          {!chatId && error && <div className="error">{error}</div>}
        </section>
      </div>
    </>
  );
}
