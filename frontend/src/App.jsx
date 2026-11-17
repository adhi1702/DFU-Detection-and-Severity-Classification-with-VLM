import { useState, useEffect, useCallback, useRef } from "react"
import { useDropzone } from "react-dropzone"
import axios from "axios"
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts"

const API = "http://localhost:8000" 

const GRADE_COLOR = {
  Mild    : { bg: "#d1fae5", text: "#065f46", border: "#6ee7b7" },
  Moderate: { bg: "#fef3c7", text: "#92400e", border: "#fcd34d" },
  Severe  : { bg: "#fee2e2", text: "#991b1b", border: "#fca5a5" },
  "N/A"   : { bg: "#f1f5f9", text: "#475569", border: "#cbd5e1" },
}

const PIE_COLORS = ["#6ee7b7", "#fcd34d", "#fca5a5"]

// ── Upload Zone ───────────────────────────────────────────
function UploadZone({ onResult, loading, setLoading }) {
  const onDrop = useCallback(async (files) => {
    if (!files[0]) return
    setLoading(true)
    const form = new FormData()
    form.append("file", files[0])
    try {
      const res = await axios.post(`${API}/predict`, form)
      onResult(res.data)
    } catch {
      alert("Prediction failed. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }, [onResult, setLoading])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "image/*": [] }, multiple: false
  })

  return (
    <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
      <input {...getInputProps()} />
      <div className="dropzone-inner">
        <div className="dropzone-icon">🩻</div>
        {loading
          ? <p className="dropzone-text">Analysing image...</p>
          : <p className="dropzone-text">
              {isDragActive ? "Drop image here" : "Drag & drop or click to upload wound image"}
            </p>
        }
        <p className="dropzone-sub">Supports JPG, PNG, BMP</p>
      </div>
    </div>
  )
}

// ── Result Panel ──────────────────────────────────────────
function ResultPanel({ result }) {
  if (!result) return (
    <div className="no-result">
      <p>📂 Upload an image to see analysis results</p>
    </div>
  )

  const gc      = GRADE_COLOR[result.severity_grade] || GRADE_COLOR["N/A"]
  const isUlcer = result.detection === "Ulcer"

  const radarData = isUlcer ? [
    { feature: "Redness",  value: Math.round((result.redness       || 0) * 100) },
    { feature: "Darkness", value: Math.round((result.darkness      || 0) * 100) },
    { feature: "Slough",   value: Math.round((result.yellow_slough || 0) * 100) },
    { feature: "Area",     value: Math.round((result.area_ratio    || 0) * 100) },
    { feature: "Texture",  value: Math.round((result.texture       || 0) * 100) },
  ] : []

  const barData = isUlcer ? [
    { name: "Redness",  value: Math.round((result.redness       || 0) * 100) },
    { name: "Darkness", value: Math.round((result.darkness      || 0) * 100) },
    { name: "Slough",   value: Math.round((result.yellow_slough || 0) * 100) },
    { name: "Area",     value: Math.round((result.area_ratio    || 0) * 100) },
    { name: "Texture",  value: Math.round((result.texture       || 0) * 100) },
  ] : []

  return (
    <div className="result-panel">
      {/* Image + detection */}
      <div className="result-top">
        <div className="result-image-wrap">
          <img
            src={`data:image/jpeg;base64,${result.image_base64}`}
            alt="wound"
            className="result-image"
          />
        </div>
        <div className="result-meta">
          <p className="result-filename">📁 {result.filename}</p>
          <div className={`detection-badge ${isUlcer ? "ulcer" : "non-ulcer"}`}>
            {isUlcer ? "🔴 Ulcer Detected" : "🟢 No Ulcer Detected"}
          </div>
          <div className="confidence-row">
            <span>Ulcer Confidence</span>
            <span className="conf-val">{result.ulcer_conf}%</span>
          </div>
          <div className="conf-bar-bg">
            <div className="conf-bar-fill" style={{ width: `${result.ulcer_conf}%` }} />
          </div>

          {isUlcer && (
            <>
              <div
                className="severity-badge"
                style={{ background: gc.bg, color: gc.text, border: `1.5px solid ${gc.border}` }}
              >
                Severity: <strong>{result.severity_grade}</strong>
                <span className="severity-score">
                  &nbsp;({(result.severity_score * 100).toFixed(1)}%)
                </span>
              </div>
              <p className="severity-desc">{result.description}</p>
              <div className="score-bar-bg">
                <div
                  className="score-bar-fill"
                  style={{
                    width: `${result.severity_score * 100}%`,
                    background:
                      result.severity_grade === "Severe"   ? "#ef4444" :
                      result.severity_grade === "Moderate" ? "#f59e0b" : "#10b981"
                  }}
                />
              </div>
            </>
          )}
          <p className="result-saved">✅ Saved to database — ID #{result.id}</p>
        </div>
      </div>

      {/* Charts */}
      {isUlcer && (
        <div className="charts-row">
          <div className="chart-box">
            <h4 className="chart-title">Feature Radar</h4>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Radar dataKey="value" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.25} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-box">
            <h4 className="chart-title">Feature Breakdown</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={barData} margin={{ left: -10 }}>
                <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: "#1e293b", border: "none", borderRadius: 8 }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Bar dataKey="value" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

// ── VLM Chat Panel ────────────────────────────────────────
function ChatPanel({ result }) {
  const [messages,  setMessages]  = useState([])
  const [input,     setInput]     = useState("")
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)
  const prevResultId = useRef(null)

  // Auto-message when new result arrives
  useEffect(() => {
    if (!result || result.id === prevResultId.current) return
    prevResultId.current = result.id

    const isUlcer = result.detection === "Ulcer"
    const opener  = isUlcer
      ? `I've analysed this wound image. The model detected a **${result.severity_grade} ulcer** with ${result.ulcer_conf}% confidence.\n\nKey findings:\n- Severity score: ${(result.severity_score * 100).toFixed(1)}%\n- Redness (inflammation): ${(result.redness * 100).toFixed(1)}%\n- Darkness (necrosis): ${(result.darkness * 100).toFixed(1)}%\n- Slough presence: ${(result.yellow_slough * 100).toFixed(1)}%\n- Wound area ratio: ${(result.area_ratio * 100).toFixed(1)}%\n\nWhat would you like to know about this wound?`
      : `I've analysed this image. The model found **no ulcer detected** (ulcer confidence: ${result.ulcer_conf}%).\n\nThe wound appears healthy or non-ulcerous. Do you have any questions about diabetic foot care or prevention?`

    setMessages([{ role: "assistant", content: opener }])
  }, [result])

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || streaming) return
    const userMsg = { role: "user", content: input.trim() }
    const updated = [...messages, userMsg]
    setMessages(updated)
    setInput("")
    setStreaming(true)

    // Add empty assistant message to stream into
    setMessages(prev => [...prev, { role: "assistant", content: "" }])

    try {
      const res = await fetch(`${API}/chat`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({
          messages       : updated,
          analysis_result: result || null,
          image_base64   : result?.image_base64 || null,
        }),
      })

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   full    = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        full += chunk
        setMessages(prev => {
          const copy = [...prev]
          copy[copy.length - 1] = { role: "assistant", content: full }
          return copy
        })
      }
    } catch {
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1] = {
          role: "assistant",
          content: "Sorry, I couldn't reach the AI. Is the backend running?"
        }
        return copy
      })
    } finally {
      setStreaming(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Simple markdown bold renderer
  const renderContent = (text) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/)
    return parts.map((p, i) =>
      p.startsWith("**") && p.endsWith("**")
        ? <strong key={i}>{p.slice(2, -2)}</strong>
        : p
    )
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-icon">🤖</span>
        <div>
          <p className="chat-title">Clinical AI Assistant</p>
          <p className="chat-sub">Powered by PodoScan</p>
        </div>
        <div className={`chat-status ${result ? "active" : ""}`}>
          {result ? "● Active" : "● Waiting"}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>🩺 Upload a wound image to start the clinical analysis conversation.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            <div className="chat-bubble">
              {msg.content
                ? msg.content.split("\n").map((line, j) => (
                    <span key={j}>
                      {renderContent(line)}
                      {j < msg.content.split("\n").length - 1 && <br />}
                    </span>
                  ))
                : <span className="typing-dots"><span /><span /><span /></span>
              }
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder={result ? "Ask about this wound..." : "Upload an image first..."}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={!result || streaming}
          rows={2}
        />
        <button
          className="chat-send"
          onClick={sendMessage}
          disabled={!input.trim() || streaming || !result}
        >
          {streaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  )
}

// ── Read-only SQL assistant drawer ───────────────────────
function SQLAssistantDrawer({ open, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const handleEscape = (event) => { if (event.key === "Escape") onClose() }
    window.addEventListener("keydown", handleEscape)
    return () => window.removeEventListener("keydown", handleEscape)
  }, [open, onClose])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages, open])

  const sendMessage = async () => {
    const question = input.trim()
    if (!question || busy) return
    const history = [...messages, { role: "user", content: question }]
    setMessages([...history, { role: "assistant", content: "" }])
    setInput("")
    setBusy(true)
    try {
      const response = await fetch(`${API}/sql-chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || "The assistant could not answer.")
      setMessages(prev => [...prev.slice(0, -1), {
        role: "assistant",
        content: payload.answer,
        table: payload.columns?.length ? { columns: payload.columns, rows: payload.rows || [] } : null,
      }])
    } catch (error) {
      setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: error.message || "Could not reach the SQL assistant." }])
    } finally {
      setBusy(false)
    }
  }

  if (!open) return null
  return (
    <div className="sql-drawer-layer">
      <button className="sql-drawer-backdrop" aria-label="Close SQL assistant" onClick={onClose} />
      <section className="sql-drawer" role="dialog" aria-modal="true" aria-labelledby="sql-drawer-title">
        <header className="sql-drawer-header">
          <span className="sql-robot" aria-hidden="true">🤖</span>
          <div className="sql-drawer-heading">
            <h2 id="sql-drawer-title">Data Assistant</h2>
            <p>Ask questions about application data</p>
          </div>
          <button className="sql-close" onClick={onClose} aria-label="Close assistant">×</button>
        </header>
        <div className="sql-messages" aria-live="polite">
          {messages.length === 0 && <div className="sql-welcome">Ask a question about the data in this application. I can only read data.</div>}
          {messages.map((message, index) => (
            <div key={index} className={`sql-message ${message.role}`}>
              <div>{message.content || (busy && index === messages.length - 1 ? "Querying the data…" : "")}</div>
              {message.table && (
                <div className="sql-result-table-wrap">
                  <table className="sql-result-table">
                    <thead><tr>{message.table.columns.map(column => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead>
                    <tbody>
                      {message.table.rows.length
                        ? message.table.rows.map((row, rowIndex) => (
                            <tr key={rowIndex}>{message.table.columns.map(column => <td key={column}>{row[column] == null ? "—" : String(row[column])}</td>)}</tr>
                          ))
                        : <tr><td colSpan={message.table.columns.length}>No rows returned.</td></tr>}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <form className="sql-composer" onSubmit={event => { event.preventDefault(); sendMessage() }}>
          <textarea
            aria-label="Ask a question about application data"
            placeholder="e.g. How many scans were severe this month?"
            value={input}
            onChange={event => setInput(event.target.value)}
            onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage() } }}
            rows={2}
            disabled={busy}
          />
          <button className="chat-send" type="submit" disabled={!input.trim() || busy}>{busy ? "…" : "Send"}</button>
        </form>
      </section>
    </div>
  )
}

// ── Stats Bar ─────────────────────────────────────────────
function StatsBar({ stats }) {
  if (!stats) return null
  const pieData = [
    { name: "Mild",     value: Number(stats.mild)     || 0 },
    { name: "Moderate", value: Number(stats.moderate) || 0 },
    { name: "Severe",   value: Number(stats.severe)   || 0 },
  ]
  return (
    <div className="stats-bar">
      <div className="stat-card">
        <p className="stat-num">{stats.total ?? 0}</p>
        <p className="stat-label">Total Scans</p>
      </div>
      <div className="stat-card">
        <p className="stat-num red">{stats.total_ulcers ?? 0}</p>
        <p className="stat-label">Ulcers</p>
      </div>
      <div className="stat-card">
        <p className="stat-num green">{stats.total_non_ulcers ?? 0}</p>
        <p className="stat-label">Non-Ulcers</p>
      </div>
      <div className="stat-card">
        <p className="stat-num yellow">{stats.avg_confidence ?? 0}%</p>
        <p className="stat-label">Avg Confidence</p>
      </div>
      <div className="stat-card pie-card">
        <PieChart width={110} height={90}>
          <Pie data={pieData} cx={50} cy={40} innerRadius={25} outerRadius={40} dataKey="value">
            {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
          </Pie>
          <Tooltip contentStyle={{ background: "#1e293b", border: "none", borderRadius: 8, fontSize: 12 }} />
        </PieChart>
        <p className="stat-label">Severity Split</p>
      </div>
    </div>
  )
}

// ── Dashboard ─────────────────────────────────────────────
function Dashboard({ predictions, onDelete, onView }) {
  return (
    <div className="dashboard">
      <h2 className="section-title">📋 Prediction History</h2>
      {predictions.length === 0
        ? <p className="empty-msg">No predictions yet. Upload an image to get started.</p>
        : (
          <div className="table-wrap">
            <table className="pred-table">
              <thead>
                <tr>
                  <th>#</th><th>File</th><th>Detection</th>
                  <th>Confidence</th><th>Severity</th>
                  <th>Score</th><th>Date</th><th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => {
                  const gc = GRADE_COLOR[p.severity_grade] || GRADE_COLOR["N/A"]
                  return (
                    <tr key={p.id}>
                      <td className="td-id">{p.id}</td>
                      <td className="td-file" title={p.filename}>{p.filename}</td>
                      <td>
                        <span className={`tbl-badge ${p.detection === "Ulcer" ? "ulcer" : "non-ulcer"}`}>
                          {p.detection}
                        </span>
                      </td>
                      <td>{p.ulcer_conf}%</td>
                      <td>
                        <span className="tbl-grade" style={{ background: gc.bg, color: gc.text }}>
                          {p.severity_grade}
                        </span>
                      </td>
                      <td>{p.severity_score != null ? (p.severity_score * 100).toFixed(1) + "%" : "—"}</td>
                      <td className="td-date">{new Date(p.created_at).toLocaleString()}</td>
                      <td className="td-actions">
                        <button className="btn-view"   onClick={() => onView(p)}>View</button>
                        <button className="btn-delete" onClick={() => onDelete(p.id)}>Delete</button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )
      }
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────
export default function App() {
  const [result,      setResult]      = useState(null)
  const [predictions, setPredictions] = useState([])
  const [stats,       setStats]       = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [tab,         setTab]         = useState("upload")
  const [sqlAssistantOpen, setSqlAssistantOpen] = useState(false)

  const fetchAll = async () => {
    try {
      const [preds, st] = await Promise.all([
        axios.get(`${API}/predictions`),
        axios.get(`${API}/stats`),
      ])
      setPredictions(preds.data)
      setStats(st.data)
    } catch {
      console.error("Could not reach backend")
    }
  }

  useEffect(() => { fetchAll() }, [])

  const handleResult = (r) => {
    setResult(r)
    fetchAll()
  }

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this prediction?")) return
    await axios.delete(`${API}/predictions/${id}`)
    fetchAll()
    if (result?.id === id) setResult(null)
  }

  const handleView = async (row) => {
    const res = await axios.get(`${API}/predictions/${row.id}`)
    setResult(res.data)
    setTab("upload")
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  return (
    <div className="app">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand">
          <img src="/foot.ico" alt="WoundSight" className="nav-icon" />
          <span>PodoScan - Scan Your Ulcered Foot</span>
        </div>
        <div className="nav-tabs">
          <button
            className={`nav-tab ${tab === "upload" ? "active" : ""}`}
            onClick={() => setTab("upload")}
          >🩻 Upload & Analyse</button>
          <button
            className={`nav-tab ${tab === "dashboard" ? "active" : ""}`}
            onClick={() => { setTab("dashboard"); fetchAll() }}
          >📋 Dashboard</button>
          <button
            className="sql-assistant-trigger"
            aria-label="Open data assistant"
            aria-expanded={sqlAssistantOpen}
            onClick={() => setSqlAssistantOpen(true)}
            title="Ask questions about app data"
          >🤖</button>
        </div>
      </nav>

      <SQLAssistantDrawer open={sqlAssistantOpen} onClose={() => setSqlAssistantOpen(false)} />

      <main className="main">
        <StatsBar stats={stats} />

        {tab === "upload" && (
          <>
            {/* Split layout */}
            <div className="split-layout">
              {/* Left — upload + results */}
              <div className="split-left">
                <h2 className="section-title">🩻 Wound Analysis</h2>
                <UploadZone
                  onResult={handleResult}
                  loading={loading}
                  setLoading={setLoading}
                />
                {loading && (
                  <div className="loader">
                    <div className="spinner" />
                    Analysing wound image...
                  </div>
                )}
                <ResultPanel result={result} />
              </div>

              {/* Right — VLM chat */}
              <div className="split-right">
                <h2 className="section-title">🤖 Clinical Assistant</h2>
                <ChatPanel result={result} />
              </div>
            </div>
          </>
        )}

        {tab === "dashboard" && (
          <Dashboard
            predictions={predictions}
            onDelete={handleDelete}
            onView={handleView}
          />
        )}
      </main>
    </div>
  )
}
