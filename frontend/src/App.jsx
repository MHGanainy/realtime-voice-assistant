import { useEffect, useRef, useState } from "react";
import "./App.css";

/**
 * Real‑time voice‑assistant development testing interface
 */
export default function App() {
  const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

  const [userTranscript, setUserTranscript] = useState("");
  const [assistantReply, setAssistantReply] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful assistant. Respond concisely and clearly."
  );
  const [isPromptLocked, setIsPromptLocked] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [conversationHistory, setConversationHistory] = useState([]);
  const [logs, setLogs] = useState([]);

  const socketRef = useRef(null);
  const recorderRef = useRef(null);
  const sendAudioRef = useRef(true);
  const expectNewAudioRef = useRef(false);

  /* ---- media‑pipeline plumbing ------------------------------------ */
  const mediaSrcRef = useRef(null);
  const sourceBufRef = useRef(null);
  const pendingChunksRef = useRef([]);
  const audioRef = useRef(null);

  /* ---- logging helper --------------------------------------------- */
  const addLog = (message, type = "info") => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }]);
  };

  /* ---- helpers ---------------------------------------------------- */
  function teardownMediaSource() {
    audioRef.current?.pause();
    if (audioRef.current?.src?.startsWith("blob:")) {
      URL.revokeObjectURL(audioRef.current.src);
    }
    mediaSrcRef.current = null;
    sourceBufRef.current = null;
    pendingChunksRef.current = [];
    audioRef.current = null;
  }

  function initMediaSource() {
    if (mediaSrcRef.current) return;

    mediaSrcRef.current = new MediaSource();
    const url = URL.createObjectURL(mediaSrcRef.current);
    audioRef.current = new Audio(url);

    audioRef.current.play().catch(() => {});

    mediaSrcRef.current.addEventListener("sourceopen", () => {
      sourceBufRef.current =
        mediaSrcRef.current.addSourceBuffer("audio/mpeg");
      sourceBufRef.current.mode = "sequence";

      sourceBufRef.current.addEventListener("updateend", flushPending);
      flushPending();
    });
  }

  function flushPending() {
    const sb = sourceBufRef.current;
    if (!sb || sb.updating) return;

    const chunk = pendingChunksRef.current.shift();
    if (!chunk) return;

    sb.appendBuffer(chunk);

    const kick = () => {
      if (audioRef.current?.paused) {
        audioRef.current.play().catch(() => {});
      }
      sb.removeEventListener("updateend", kick);
    };
    sb.addEventListener("updateend", kick);
  }

  const sendCommand = (command, data = {}) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ command, ...data }));
      addLog(`Sent command: ${command}`, "command");
    }
  };

  const startRecording = async () => {
    setConnectionStatus("connecting");
    addLog("Connecting to WebSocket...", "info");
    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = async () => {
      setConnectionStatus("connected");
      addLog("WebSocket connected", "success");
      sendCommand("set_prompt", { prompt: systemPrompt });
      setIsPromptLocked(true);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });
        recorderRef.current = recorder;
        sendAudioRef.current = true;

        recorder.ondataavailable = async (e) => {
          if (
            !e.data.size ||
            socket.readyState !== WebSocket.OPEN ||
            !sendAudioRef.current
          )
            return;
          const buf = await e.data.arrayBuffer();
          socket.send(buf);
        };

        recorder.start(250);
        setIsRecording(true);
        addLog("Recording started", "success");
      } catch (err) {
        console.error("Mic access denied:", err);
        addLog(`Mic access error: ${err.message}`, "error");
        socket.close();
        setConnectionStatus("error");
      }
    };

    socket.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);

        if (msg.type === "interaction_complete") {
          setUserTranscript(msg.utterance);
          setAssistantReply(msg.response);
          setConversationHistory(prev => [
            ...prev,
            { role: "user", content: msg.utterance, timestamp: new Date() },
            { role: "assistant", content: msg.response, timestamp: new Date() }
          ]);
          addLog("Interaction complete", "success");
          return;
        }

        if (msg.command === "pause") {
          sendAudioRef.current = false;
          expectNewAudioRef.current = true;
          recorderRef.current?.pause();
          addLog("Audio paused by backend", "info");
          return;
        }
        if (msg.command === "resume") {
          sendAudioRef.current = true;
          recorderRef.current?.resume?.();
          addLog("Audio resumed by backend", "info");
          return;
        }

        if (msg.transcript) {
          setUserTranscript(msg.transcript);
          if (!msg.final) {
            addLog(`Partial transcript: "${msg.transcript}"`, "transcript");
          }
        }

        if (msg.type === "command_response") {
          addLog(`Command response: ${JSON.stringify(msg)}`, "response");
        }
      } else {
        if (expectNewAudioRef.current) {
          teardownMediaSource();
          expectNewAudioRef.current = false;
          addLog("Starting new audio stream", "info");
        }

        initMediaSource();
        pendingChunksRef.current.push(new Uint8Array(e.data));
        flushPending();
      }
    };

    socket.onerror = (err) => {
      console.error("WebSocket error", err);
      addLog("WebSocket error", "error");
      setConnectionStatus("error");
    };
    
    socket.onclose = () => {
      addLog("WebSocket disconnected", "info");
      stopRecording();
      setIsPromptLocked(false);
      setConnectionStatus("disconnected");
    };
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    socketRef.current?.close();
    setIsRecording(false);
    setUserTranscript("");
    setAssistantReply("");
    teardownMediaSource();
    setIsPromptLocked(false);
    setConnectionStatus("disconnected");
    addLog("Recording stopped", "info");
  };

  const handlePromptChange = (e) => {
    setSystemPrompt(e.target.value);
  };

  const handleUpdatePrompt = () => {
    if (isRecording) {
      sendCommand("clear_history");
      sendCommand("set_prompt", { prompt: systemPrompt });
      setUserTranscript("");
      setAssistantReply("");
      setConversationHistory([]);
      addLog("Prompt updated and history cleared", "success");
    }
  };

  const clearLogs = () => {
    setLogs([]);
    addLog("Logs cleared", "info");
  };

  const clearHistory = () => {
    setConversationHistory([]);
    if (isRecording) {
      sendCommand("clear_history");
    }
    addLog("History cleared", "info");
  };

  useEffect(() => () => stopRecording(), []);

  /* ---- UI --------------------------------------------------------- */
  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Voice Assistant Dev Testing</h1>
        <div className={`connection-status ${connectionStatus}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {connectionStatus === "connected" ? "Connected" : 
             connectionStatus === "connecting" ? "Connecting..." : 
             connectionStatus === "error" ? "Error" : "Disconnected"}
          </span>
        </div>
      </header>

      <div className="app-layout">
        {/* Left Panel - Controls & Current Interaction */}
        <div className="left-panel">
          {/* System Prompt Section */}
          <section className="prompt-section">
            <div className="section-header">
              <h2>System Prompt</h2>
              {isPromptLocked && (
                <span className="lock-indicator">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/>
                  </svg>
                  Locked
                </span>
              )}
            </div>
            <textarea
              value={systemPrompt}
              onChange={handlePromptChange}
              disabled={isPromptLocked && !isRecording}
              placeholder="Enter system prompt..."
              className={`prompt-input ${isPromptLocked ? 'locked' : ''}`}
              rows="4"
            />
            {isRecording && (
              <button onClick={handleUpdatePrompt} className="update-button">
                Update & Clear History
              </button>
            )}
          </section>

          {/* Control Section */}
          <div className="control-section">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`record-button ${isRecording ? 'recording' : ''}`}
            >
              <div className="button-content">
                <svg className="mic-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="3" width="6" height="11" rx="3"/>
                  <path d="M5 12v1a7 7 0 0014 0v-1M12 18v3"/>
                </svg>
                <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
                {isRecording && <span className="recording-indicator"></span>}
              </div>
            </button>
          </div>

          {/* Current Interaction */}
          <section className="current-interaction">
            <h3>Current Interaction</h3>
            <div className="interaction-messages">
              <div className="message user-message">
                <div className="message-label">USER</div>
                <div className="message-content">
                  {userTranscript || <span className="placeholder">Waiting for speech...</span>}
                </div>
              </div>
              <div className="message assistant-message">
                <div className="message-label">ASSISTANT</div>
                <div className="message-content">
                  {assistantReply || <span className="placeholder">Waiting for response...</span>}
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Middle Panel - Conversation History */}
        <div className="middle-panel">
          <section className="history-section">
            <div className="section-header">
              <h2>Conversation History</h2>
              <button onClick={clearHistory} className="clear-button">Clear</button>
            </div>
            <div className="history-container">
              {conversationHistory.length === 0 ? (
                <div className="empty-state">No conversation history yet</div>
              ) : (
                conversationHistory.map((msg, idx) => (
                  <div key={idx} className={`history-message ${msg.role}`}>
                    <div className="history-header">
                      <span className="history-label">{msg.role.toUpperCase()}</span>
                      <span className="history-time">
                        {msg.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="history-content">{msg.content}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        {/* Right Panel - Debug Logs */}
        <div className="right-panel">
          <section className="logs-section">
            <div className="section-header">
              <h2>Debug Logs</h2>
              <button onClick={clearLogs} className="clear-button">Clear</button>
            </div>
            <div className="logs-container">
              {logs.map((log, idx) => (
                <div key={idx} className={`log-entry ${log.type}`}>
                  <span className="log-time">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}