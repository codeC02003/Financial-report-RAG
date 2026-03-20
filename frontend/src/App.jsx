import { useState, useRef, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  FiUploadCloud, FiMessageSquare, FiEye,
  FiSend, FiX, FiChevronDown, FiCpu, FiLayers,
  FiZap, FiSearch, FiBookOpen, FiArrowRight, FiCheck,
  FiGlobe, FiShield, FiTrendingUp, FiArrowLeft, FiFile,
  FiGithub, FiMail, FiExternalLink, FiChevronUp, FiDatabase,
  FiBarChart2, FiAlertTriangle, FiFileText, FiChevronRight
} from 'react-icons/fi';
import { uploadDocument, askQuestion, getPageImage } from './api';
import './App.css';

// ─── Method badge config ───
const METHOD_CONFIG = {
  table:          { label: 'Table Extraction',  icon: '📊', color: '#10b981', bg: 'rgba(16,185,129,0.12)' },
  extractive:     { label: 'Extractive QA',     icon: '🔍', color: '#3b82f6', bg: 'rgba(59,130,246,0.12)' },
  vision:         { label: 'Vision AI',          icon: '👁', color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
  generative:     { label: 'Generative AI',     icon: '🤖', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  conversational: { label: 'Conversational',     icon: '💬', color: '#f43f5e', bg: 'rgba(244,63,94,0.12)' },
  unanswerable:   { label: 'Unanswerable',       icon: '❓', color: '#6b7280', bg: 'rgba(107,114,128,0.12)' },
};

// ─── Question categories ───
const QUESTION_CATEGORIES = {
  Facts:      ['What was total revenue?', 'Who is the CEO?', 'What is the fiscal year end?'],
  Trends:     ['How did revenue change over the years?', 'When did net income increase?', 'What are the growth trends?'],
  Comparison: ['Compare 2024 vs 2023 revenue', 'Difference in operating expenses', 'Compare revenue across segments'],
  Risks:      ['What are the main risks?', 'What legal proceedings exist?', 'What are the major risk factors?'],
  Summary:    ['Summarize financial performance', 'What is this document about?', 'Key financial highlights'],
};

// ─── Answer rendering (supports bold, bullets, line breaks) ───
function renderAnswer(text) {
  if (!text) return null;
  const lines = text.split('\n');
  const elements = [];
  let bulletGroup = [];
  let key = 0;

  const flushBullets = () => {
    if (bulletGroup.length > 0) {
      elements.push(
        <ul key={`ul-${key++}`} className="answer-list">
          {bulletGroup.map((b, i) => <li key={i}>{formatInline(b)}</li>)}
        </ul>
      );
      bulletGroup = [];
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { flushBullets(); continue; }
    if (/^[-•*]\s+/.test(trimmed)) {
      bulletGroup.push(trimmed.replace(/^[-•*]\s+/, ''));
    } else {
      flushBullets();
      elements.push(<p key={`p-${key++}`} className="answer-para">{formatInline(trimmed)}</p>);
    }
  }
  flushBullets();
  return elements.length > 0 ? elements : text;
}

function formatInline(text) {
  const parts = text.split(/(\*\*.+?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="answer-bold">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

// ─── Loading text component (4-stage) ───
function LoadingIndicator() {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 1500);
    const t2 = setTimeout(() => setPhase(2), 4000);
    const t3 = setTimeout(() => setPhase(3), 8000);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, []);
  const stages = [
    'Searching document...',
    'Reranking results...',
    'Analyzing context...',
    'Generating answer...',
  ];
  return (
    <div className="loading-indicator-row">
      <div className="typing-dots"><span /><span /><span /></div>
      <div className="loading-stage-info">
        <span className="loading-status-text">{stages[phase]}</span>
        <div className="loading-progress">
          {stages.map((s, i) => (
            <div key={i} className={`loading-progress-dot ${i <= phase ? 'active' : ''}`} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Evidence panel component ───
function EvidencePanel({ evidence }) {
  const [open, setOpen] = useState(false);
  if (!evidence || evidence.length === 0) return null;
  const top3 = evidence.slice(0, 3);
  return (
    <div className="evidence-section">
      <button className="evidence-toggle" onClick={() => setOpen(!open)}>
        {open ? <FiChevronUp /> : <FiChevronDown />} {open ? 'Hide' : 'Show'} Evidence ({evidence.length})
      </button>
      {open && (
        <div className="evidence-list">
          {top3.map((ev, i) => (
            <div key={ev.chunk_id || i} className="evidence-card">
              <div className="evidence-header">
                <span className="evidence-page">Page {ev.page}</span>
                <span className="evidence-type">{ev.type || 'text'}</span>
                <span className="evidence-score">{Math.round((ev.score || 0) * 100)}% relevance</span>
              </div>
              <p className="evidence-text">{ev.text?.slice(0, 200)}{ev.text?.length > 200 ? '...' : ''}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Follow-up chips ───
function FollowUpChips({ followUps, onAsk, disabled }) {
  if (!followUps || followUps.length === 0) return null;
  return (
    <div className="followup-section">
      <span className="followup-label">Ask next:</span>
      <div className="followup-chips">
        {followUps.map((q) => (
          <button key={q} className="followup-chip" onClick={() => onAsk(q)} disabled={disabled}>
            <FiChevronRight className="followup-icon" /> {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [docInfo, setDocInfo] = useState(null);
  const [currentPage, setCurrentPage] = useState(null);
  const [currentPageImage, setCurrentPageImage] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isLoadingPage, setIsLoadingPage] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('hero');
  const [showPageViewer, setShowPageViewer] = useState(false);
  const [activeCategory, setActiveCategory] = useState('Facts');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const heroRef = useRef(null);
  const featuresRef = useRef(null);
  const aboutRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (chatOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [chatOpen]);

  // Scroll spy
  useEffect(() => {
    if (chatOpen) return;
    const handleScroll = () => {
      const scrollY = window.scrollY + 100;
      if (aboutRef.current && scrollY >= aboutRef.current.offsetTop) {
        setActiveSection('about');
      } else if (featuresRef.current && scrollY >= featuresRef.current.offsetTop) {
        setActiveSection('features');
      } else {
        setActiveSection('hero');
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [chatOpen]);

  const scrollTo = (ref) => {
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadPage = useCallback(async (pageNumber) => {
    setCurrentPage(pageNumber);
    setIsLoadingPage(true);
    try {
      const data = await getPageImage(pageNumber);
      setCurrentPageImage(data.image);
    } catch {
      setCurrentPageImage(null);
    } finally {
      setIsLoadingPage(false);
    }
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) return;

    setIsUploading(true);
    setMessages([]);
    setCurrentPage(null);
    setCurrentPageImage(null);

    try {
      const info = await uploadDocument(file);
      setDocInfo(info);
      if (info.pages_processed > 0) loadPage(1);
      setMessages([{
        type: 'system',
        pipeline: true,
        pages: info.pages_processed || 0,
        tables: info.pages_with_tables || 0,
        charts: info.pages_with_charts || 0,
        chunks: info.total_chunks || 0,
      }]);
    } catch (err) {
      setMessages([{
        type: 'system',
        text: 'Failed to upload: ' + (err.response?.data?.detail || err.message)
      }]);
    } finally {
      setIsUploading(false);
    }
  }, [loadPage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
  });

  // Build conversation history from messages
  const buildHistory = () => {
    const history = [];
    for (const msg of messages) {
      if (msg.type === 'user') {
        history.push({ role: 'user', content: msg.text });
      } else if (msg.type === 'answer' && msg.data) {
        history.push({ role: 'assistant', content: msg.data.answer });
      }
    }
    return history;
  };

  const doAsk = async (q) => {
    if (!q || isAsking || !docInfo) return;
    setQuestion('');
    setIsAsking(true);

    const history = buildHistory();

    setMessages((prev) => [...prev, { type: 'user', text: q }]);
    setMessages((prev) => [...prev, { type: 'loading' }]);

    try {
      const result = await askQuestion(q, 8, history);
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.type !== 'loading');
        return [...filtered, { type: 'answer', data: result }];
      });
      if (result.source_pages?.length > 0) {
        loadPage(result.source_pages[0]);
      }
    } catch (err) {
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.type !== 'loading');
        return [...filtered, {
          type: 'answer',
          data: {
            answer: 'Something went wrong. Please try again.',
            confidence: 0,
            is_unanswerable: true,
            source_pages: [],
            evidence: [],
            used_vision: false,
            method: 'unanswerable',
            follow_ups: [],
          }
        }];
      });
    } finally {
      setIsAsking(false);
      inputRef.current?.focus();
    }
  };

  const handleAsk = () => {
    doAsk(question.trim());
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const getConfidenceClass = (c) => {
    if (c >= 0.7) return 'conf-high';
    if (c >= 0.4) return 'conf-med';
    return 'conf-low';
  };

  const getMethodBadge = (d) => {
    const method = d.method || (d.is_unanswerable ? 'unanswerable' : d.used_vision ? 'vision' : 'generative');
    const cfg = METHOD_CONFIG[method] || METHOD_CONFIG.generative;
    return (
      <span className="method-badge" style={{ color: cfg.color, background: cfg.bg }}>
        {cfg.icon} {cfg.label}
      </span>
    );
  };

  const openChat = () => setChatOpen(true);
  const closeChat = () => { setChatOpen(false); setShowPageViewer(false); };

  // ─── Pipeline status card component ───
  const PipelineCard = ({ pages, tables, charts, chunks }) => (
    <div className="pipeline-card">
      <div className="pipeline-row done"><FiCheck className="pipeline-icon success" /> PDF uploaded successfully</div>
      <div className="pipeline-row done"><FiCheck className="pipeline-icon success" /> Extracted text ({pages} pages)</div>
      <div className="pipeline-row done"><FiCheck className="pipeline-icon success" /> Detected {tables} tables, {charts} charts</div>
      <div className="pipeline-row done"><FiCheck className="pipeline-icon success" /> Built search index ({chunks} chunks)</div>
      <div className="pipeline-row ready">Ready for questions!</div>
    </div>
  );

  // ─── Page navigator ───
  const PageNavigator = () => {
    if (!docInfo || !currentPage) return null;
    return (
      <div className="page-nav">
        <button
          className="page-nav-btn"
          onClick={() => { loadPage(Math.max(1, currentPage - 1)); setShowPageViewer(true); }}
          disabled={currentPage <= 1}
        >
          <FiArrowLeft />
        </button>
        <button className="page-nav-current" onClick={() => setShowPageViewer(true)}>
          <FiFile /> Page {currentPage} / {docInfo.pages_processed}
        </button>
        <button
          className="page-nav-btn"
          onClick={() => { loadPage(Math.min(docInfo.pages_processed, currentPage + 1)); setShowPageViewer(true); }}
          disabled={currentPage >= docInfo.pages_processed}
        >
          <FiArrowRight />
        </button>
      </div>
    );
  };

  // ─── Fullscreen Chat View ───
  if (chatOpen) {
    return (
      <div className="chat-fullscreen">
        {/* Page viewer overlay */}
        {showPageViewer && (
          <div className="page-overlay">
            <div className="page-overlay-header">
              <button onClick={() => setShowPageViewer(false)} className="page-overlay-back">
                <FiArrowLeft /> Back to Chat
              </button>
              <span className="page-overlay-title">Page {currentPage} of {docInfo?.pages_processed || '?'}</span>
              <div className="page-overlay-nav">
                <button
                  onClick={() => loadPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage <= 1}
                  className="page-nav-btn-sm"
                >
                  <FiArrowLeft />
                </button>
                <button
                  onClick={() => loadPage(Math.min(docInfo?.pages_processed || currentPage, currentPage + 1))}
                  disabled={currentPage >= (docInfo?.pages_processed || currentPage)}
                  className="page-nav-btn-sm"
                >
                  <FiArrowRight />
                </button>
              </div>
              <button onClick={() => setShowPageViewer(false)} className="page-overlay-close"><FiX /></button>
            </div>
            <div className="page-overlay-body">
              {isLoadingPage ? (
                <div className="page-overlay-loading"><div className="loading-spinner large" /></div>
              ) : currentPageImage ? (
                <img src={`data:image/png;base64,${currentPageImage}`} alt={`Page ${currentPage}`} />
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>No page to display</p>
              )}
            </div>
          </div>
        )}

        {/* Chat header */}
        <div className="chat-fs-header">
          <button className="chat-fs-back" onClick={closeChat}>
            <FiArrowLeft />
          </button>
          <div className="chat-fs-brand">
            <div className="chat-fs-avatar"><FiZap /></div>
            <div>
              <div className="chat-fs-title">FinRAG Assistant</div>
              <div className="chat-fs-status">
                {docInfo ? (
                  <>
                    <span className="status-dot online" />
                    {docInfo.pages_processed} pages &middot; {docInfo.pages_with_tables || 0} tables &middot; {docInfo.pages_with_charts || 0} charts &middot; {docInfo.total_chunks} chunks
                  </>
                ) : (
                  <><span className="status-dot" /> Upload a PDF to start</>
                )}
              </div>
            </div>
          </div>
          <PageNavigator />
        </div>

        {/* Chat body */}
        <div className="chat-fs-body">
          {/* Upload prompt */}
          {!docInfo && !isUploading && (
            <div className="chat-upload-area">
              <div {...getRootProps()} className={`chat-dropzone-full ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="chat-drop-icon"><FiUploadCloud /></div>
                <h3>Upload a document</h3>
                <p>Drop a PDF here or click to browse</p>
              </div>
            </div>
          )}

          {isUploading && (
            <div className="chat-upload-area">
              <div className="loading-spinner large" />
              <p style={{ marginTop: 16, color: 'var(--text-muted)', fontWeight: 500 }}>Processing document...</p>
            </div>
          )}

          {/* Messages */}
          <div className="chat-fs-messages">
            {messages.map((msg, idx) => {
              if (msg.type === 'system') {
                return (
                  <div key={idx} className="msg msg-system">
                    {msg.pipeline ? (
                      <PipelineCard pages={msg.pages} tables={msg.tables} charts={msg.charts} chunks={msg.chunks} />
                    ) : (
                      <div className="msg-system-pill">{msg.text}</div>
                    )}
                  </div>
                );
              }
              if (msg.type === 'user') {
                return (
                  <div key={idx} className="msg msg-user">
                    <div className="msg-bubble user-bubble">{msg.text}</div>
                  </div>
                );
              }
              if (msg.type === 'loading') {
                return (
                  <div key={idx} className="msg msg-bot">
                    <div className="msg-bot-avatar"><FiZap /></div>
                    <div className="msg-bubble bot-bubble">
                      <LoadingIndicator />
                    </div>
                  </div>
                );
              }
              if (msg.type === 'answer') {
                const d = msg.data;
                return (
                  <div key={idx} className="msg msg-bot">
                    <div className="msg-bot-avatar"><FiZap /></div>
                    <div className="msg-bot-content">
                      <div className={`msg-bubble bot-bubble ${d.is_unanswerable ? 'unanswerable' : ''}`}>
                        <div className="answer-content">
                          {renderAnswer(d.answer)}
                        </div>
                      </div>
                      <div className="msg-meta">
                        {getMethodBadge(d)}
                        <span className={`msg-conf ${getConfidenceClass(d.confidence)}`}>
                          {Math.round(d.confidence * 100)}%
                        </span>
                        {d.used_vision && <span className="msg-tag vision-tag"><FiEye /> Vision</span>}
                        {d.source_pages?.length > 0 && (
                          <span className="msg-sources">
                            {d.source_pages.slice(0, 5).map((p) => (
                              <button key={p} onClick={() => { loadPage(p); setShowPageViewer(true); }} className="msg-page-btn">
                                p.{p}
                              </button>
                            ))}
                          </span>
                        )}
                      </div>
                      <EvidencePanel evidence={d.evidence} />
                      <FollowUpChips followUps={d.follow_ups} onAsk={doAsk} disabled={isAsking} />
                    </div>
                  </div>
                );
              }
              return null;
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Chat input */}
        <div className="chat-fs-footer">
          {docInfo && (
            <>
              {/* Question type selector */}
              <div className="category-row">
                {Object.keys(QUESTION_CATEGORIES).map((cat) => (
                  <button
                    key={cat}
                    className={`category-pill ${activeCategory === cat ? 'active' : ''}`}
                    onClick={() => setActiveCategory(cat)}
                  >
                    {cat === 'Facts' && <FiDatabase />}
                    {cat === 'Trends' && <FiTrendingUp />}
                    {cat === 'Comparison' && <FiBarChart2 />}
                    {cat === 'Risks' && <FiAlertTriangle />}
                    {cat === 'Summary' && <FiFileText />}
                    {cat}
                  </button>
                ))}
              </div>
              <div className="chat-quick-row">
                {QUESTION_CATEGORIES[activeCategory].map((q) => (
                  <button key={q} className="quick-chip" onClick={() => doAsk(q)} disabled={isAsking}>
                    {q}
                  </button>
                ))}
              </div>
              <div className="chat-input-row">
                <input
                  ref={inputRef}
                  type="text"
                  className="chat-input"
                  placeholder="Ask anything about the document..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isAsking}
                />
                <button
                  className="chat-send-btn"
                  onClick={handleAsk}
                  disabled={isAsking || !question.trim()}
                >
                  <FiSend />
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  // ─── Landing Page ───
  return (
    <div className="site">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-inner">
          <div className="nav-brand" onClick={() => scrollTo(heroRef)}>
            <div className="nav-logo"><FiZap /></div>
            <span>FinRAG</span>
          </div>
          <div className="nav-links">
            <button className={activeSection === 'hero' ? 'active' : ''} onClick={() => scrollTo(heroRef)}>Home</button>
            <button className={activeSection === 'features' ? 'active' : ''} onClick={() => scrollTo(featuresRef)}>Features</button>
            <button className={activeSection === 'about' ? 'active' : ''} onClick={() => scrollTo(aboutRef)}>About</button>
            <button className="nav-cta" onClick={openChat}>
              <FiMessageSquare /> Try It
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero" ref={heroRef}>
        <div className="hero-bg-orb orb-1" />
        <div className="hero-bg-orb orb-2" />
        <div className="hero-bg-orb orb-3" />
        <div className="hero-content">
          <div className="hero-badge"><FiCpu /> Purpose-Built for Financial Documents</div>
          <h1>Analyze Annual Reports with AI<br /><span className="gradient-text">Get Verified Answers with Citations</span></h1>
          <p className="hero-sub">
            Hybrid retrieval (BM25 + semantic search), table-aware extraction, page-level citations,
            and multimodal reasoning — purpose-built for financial documents.
          </p>
          <div className="hero-capability-badges">
            <span className="capability-badge">Hybrid RAG</span>
            <span className="capability-badge">Table-Aware QA</span>
            <span className="capability-badge">Financial Reports</span>
            <span className="capability-badge">Multimodal AI</span>
          </div>
          <div className="hero-actions">
            <button className="btn-primary" onClick={openChat}>
              Start Asking <FiArrowRight />
            </button>
            <button className="btn-secondary" onClick={() => scrollTo(featuresRef)}>
              Learn More <FiChevronDown />
            </button>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-num">100%</span>
              <span className="hero-stat-label">Accuracy (53 Questions)</span>
            </div>
            <div className="hero-stat-divider" />
            <div className="hero-stat">
              <span className="hero-stat-num">4</span>
              <span className="hero-stat-label">AI Models</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features" ref={featuresRef}>
        <div className="section-inner">
          <div className="section-label"><FiLayers /> Capabilities</div>
          <h2 className="section-title">Everything you need to<br /><span className="gradient-text">understand your documents</span></h2>
          <div className="features-grid">
            {[
              { icon: <FiSearch />, color: '#60a5fa', bg: 'rgba(59,130,246,0.12)', title: 'Hybrid RAG Retrieval', desc: 'BM25 keyword search combined with semantic embeddings and cross-encoder reranking — finds the most relevant chunks every time.' },
              { icon: <FiMessageSquare />, color: '#f472b6', bg: 'rgba(244,114,182,0.12)', title: 'Conversational AI', desc: 'Ask follow-up questions naturally. The system remembers your conversation and suggests smart next questions.' },
              { icon: <FiEye />, color: '#34d399', bg: 'rgba(52,211,153,0.12)', title: 'Vision Understanding', desc: 'Reads charts, graphs, and visual elements using multimodal AI. Understands what your eyes see on the page.' },
              { icon: <FiShield />, color: '#fbbf24', bg: 'rgba(251,191,36,0.12)', title: 'Unanswerable Detection', desc: 'Knows when to say "I don\'t know". If the document doesn\'t contain the answer, it tells you honestly.' },
              { icon: <FiTrendingUp />, color: '#22d3ee', bg: 'rgba(34,211,238,0.12)', title: 'Table-Aware Financial QA', desc: 'Precisely extracts data from financial tables, computes comparisons, trends, and percentages from actual document data.' },
              { icon: <FiGlobe />, color: '#a78bfa', bg: 'rgba(167,139,250,0.12)', title: 'Any Financial Document', desc: 'Works with 10-K filings, annual reports, earnings calls, research papers, and any PDF document you throw at it.' },
            ].map((f) => (
              <div key={f.title} className="feature-card">
                <div className="feature-icon" style={{ background: f.bg, color: f.color }}>{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About */}
      <section className="about" ref={aboutRef}>
        <div className="section-inner">
          <div className="about-grid">
            <div className="about-text">
              <div className="section-label"><FiBookOpen /> About</div>
              <h2 className="section-title">Built with a multi-stage<br /><span className="gradient-text">AI pipeline</span></h2>
              <p className="about-desc">
                FinRAG uses a sophisticated multi-stage architecture combining the best of
                retrieval, extraction, and generation to deliver accurate answers every time.
              </p>
              <div className="about-steps">
                {[
                  { n: '1', title: 'Hybrid Retrieval', desc: 'BM25 keyword search + semantic embeddings + cross-encoder reranking to find the most relevant chunks.' },
                  { n: '2', title: 'Table Extraction', desc: 'High-precision extractors for financial tables, comparisons, percentages, and trends — zero hallucination.' },
                  { n: '3', title: 'LLM Elaboration', desc: 'Extracted facts are enriched by Qwen2-VL into detailed, document-quality answers with full context.' },
                  { n: '4', title: 'Smart Follow-ups', desc: 'Every answer comes with contextual follow-up suggestions to help you explore the document deeper.' },
                ].map((s) => (
                  <div key={s.n} className="about-step">
                    <div className="step-num">{s.n}</div>
                    <div><strong>{s.title}</strong><span>{s.desc}</span></div>
                  </div>
                ))}
              </div>
            </div>
            <div className="about-visual">
              <div className="tech-stack">
                <h4>Tech Stack</h4>
                <div className="tech-tags">
                  {['PyTorch', 'Transformers', 'FAISS', 'FastAPI', 'React', 'RoBERTa', 'Qwen2-VL', 'Sentence-Transformers', 'PyMuPDF', 'pdfplumber'].map(t => (
                    <span key={t} className="tech-tag">{t}</span>
                  ))}
                </div>
                <div className="tech-highlights">
                  {[
                    'Zero hallucination on verified data',
                    'Runs locally — no data leaves your machine',
                    'Handles 100+ page documents',
                    'Smart follow-up suggestions',
                    'Contextual page navigation',
                  ].map(t => (
                    <div key={t} className="tech-hl"><FiCheck /> <span>{t}</span></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="section-inner">
          <div className="footer-inner">
            <div className="footer-brand"><FiZap /> FinRAG</div>
            <p>Multimodal Document Question Answering System</p>
            <div className="footer-links">
              <a href="https://my-portfolio-mu-ten-24.vercel.app/" target="_blank" rel="noopener noreferrer" className="footer-link">
                <FiExternalLink /> Portfolio
              </a>
              <a href="https://github.com/codeC02003" target="_blank" rel="noopener noreferrer" className="footer-link">
                <FiGithub /> GitHub
              </a>
              <a href="mailto:chinmaymhatre@arizona.edu" className="footer-link">
                <FiMail /> Contact
              </a>
            </div>
            <div className="footer-divider" />
            <p className="footer-copy">Built by Chinmay Mhatre</p>
          </div>
        </div>
      </footer>

      {/* Floating FAB */}
      <button className="chat-fab" onClick={openChat}>
        <FiMessageSquare />
        {docInfo && <span className="fab-badge" />}
      </button>
    </div>
  );
}

export default App;
