import { useState, useRef, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  FiUploadCloud, FiMessageSquare, FiEye,
  FiSend, FiX, FiChevronDown, FiCpu, FiLayers,
  FiZap, FiSearch, FiBookOpen, FiArrowRight, FiCheck,
  FiGlobe, FiShield, FiTrendingUp, FiArrowLeft, FiFile,
  FiGithub, FiMail, FiExternalLink
} from 'react-icons/fi';
import { uploadDocument, askQuestion, getPageImage } from './api';
import './App.css';

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
        text: `Document loaded! ${info.pages_processed} pages processed and ${info.total_chunks} chunks indexed. I'm ready to answer your questions.`
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
            used_vision: false,
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

  const openChat = () => setChatOpen(true);
  const closeChat = () => { setChatOpen(false); setShowPageViewer(false); };

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
              <span className="page-overlay-title">Page {currentPage}</span>
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
                  <><span className="status-dot online" /> {docInfo.pages_processed} pages &middot; {docInfo.total_chunks} chunks</>
                ) : (
                  <><span className="status-dot" /> Upload a PDF to start</>
                )}
              </div>
            </div>
          </div>
          {currentPageImage && (
            <button className="chat-fs-page-btn" onClick={() => setShowPageViewer(true)}>
              <FiFile /> Page {currentPage}
            </button>
          )}
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
                    <div className="msg-system-pill">{msg.text}</div>
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
                      <div className="typing-dots"><span /><span /><span /></div>
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
                        {d.answer}
                      </div>
                      <div className="msg-meta">
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
              <div className="chat-quick-row">
                {['What is this document about?', 'Key financial highlights?', 'What are the main risks?', 'Who is the CEO?'].map((q) => (
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
          <div className="hero-badge"><FiCpu /> Powered by AI</div>
          <h1>Ask your documents<br /><span className="gradient-text">anything.</span></h1>
          <p className="hero-sub">
            Upload any PDF and get instant, accurate answers. From financial reports
            to research papers — our AI reads, understands, and responds with precision.
          </p>
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
              <span className="hero-stat-num">94%</span>
              <span className="hero-stat-label">Accuracy (50 Questions)</span>
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
              { icon: <FiSearch />, color: '#60a5fa', bg: 'rgba(59,130,246,0.12)', title: 'Smart Extraction', desc: 'Precisely extracts financial data, tables, and specific values. No hallucination — answers come directly from your document.' },
              { icon: <FiMessageSquare />, color: '#f472b6', bg: 'rgba(244,114,182,0.12)', title: 'Conversational AI', desc: 'Ask follow-up questions naturally. "What about Services?" after asking about iPhone — it remembers the conversation.' },
              { icon: <FiEye />, color: '#34d399', bg: 'rgba(52,211,153,0.12)', title: 'Vision Understanding', desc: 'Reads charts, graphs, and visual elements using multimodal AI. Understands what your eyes see on the page.' },
              { icon: <FiShield />, color: '#fbbf24', bg: 'rgba(251,191,36,0.12)', title: 'Unanswerable Detection', desc: 'Knows when to say "I don\'t know". If the document doesn\'t contain the answer, it tells you honestly.' },
              { icon: <FiTrendingUp />, color: '#22d3ee', bg: 'rgba(34,211,238,0.12)', title: 'Comparison & Analysis', desc: 'Compare data across years, calculate percentages, and identify trends — all computed from actual document data.' },
              { icon: <FiGlobe />, color: '#a78bfa', bg: 'rgba(167,139,250,0.12)', title: 'Any Document Type', desc: 'Works with 10-K filings, research papers, contracts, reports, and any PDF document you throw at it.' },
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
                  { n: '2', title: 'Pattern Extraction', desc: 'High-precision extractors for financial tables, comparisons, percentages, and specific facts.' },
                  { n: '3', title: 'Extractive QA', desc: 'RoBERTa-SQuAD2 for precise span extraction — no hallucination, just exact text from the document.' },
                  { n: '4', title: 'Generative Fallback', desc: 'Qwen2-VL for conversational answers and visual understanding when precise extraction isn\'t enough.' },
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
                    'Conversational follow-up questions',
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
