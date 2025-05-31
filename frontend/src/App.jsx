import { useState, useRef, useEffect } from 'react';
import protobuf from 'protobufjs';
import './App.css';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don\'t include special characters in your answers.');
  const [tempSystemPrompt, setTempSystemPrompt] = useState('You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don\'t include special characters in your answers.');
  const [isPromptLocked, setIsPromptLocked] = useState(true);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [logs, setLogs] = useState([]);
  const [currentInteraction, setCurrentInteraction] = useState({ user: '', assistant: '' });
  const [latencyData, setLatencyData] = useState({ llm: 0, tts: 0, total: 0 });
  
  const wsRef = useRef(null);
  const dataWsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const sourceRef = useRef(null);
  const frameTypeRef = useRef(null);
  const playTimeRef = useRef(0);
  const lastMessageTimeRef = useRef(0);
  const logsEndRef = useRef(null);
  const conversationEndRef = useRef(null);
  const startTimeRef = useRef(null);
  const audioChunkCountRef = useRef(0);
  
  // Constants
  const SAMPLE_RATE = 16000;
  const NUM_CHANNELS = 1;
  const PLAY_TIME_RESET_THRESHOLD_MS = 1.0;
  
  // Load protobuf schema
  useEffect(() => {
    // Skip if already loaded (React StrictMode runs effects twice)
    if (frameTypeRef.current) return;
    
    const protoDefinition = `
      syntax = "proto3";
      package pipecat;
      
      message TextFrame {
        uint64 id = 1;
        string name = 2;
        string text = 3;
      }
      
      message AudioRawFrame {
        uint64 id = 1;
        string name = 2;
        bytes audio = 3;
        uint32 sample_rate = 4;
        uint32 num_channels = 5;
        optional uint64 pts = 6;
      }
      
      message TranscriptionFrame {
        uint64 id = 1;
        string name = 2;
        string text = 3;
        string user_id = 4;
        string timestamp = 5;
      }
      
      message Frame {
        oneof frame {
          TextFrame text = 1;
          AudioRawFrame audio = 2;
          TranscriptionFrame transcription = 3;
        }
      }
    `;
    
    try {
      const root = protobuf.parse(protoDefinition).root;
      frameTypeRef.current = root.lookupType('pipecat.Frame');
      addLog('info', 'Protobuf schema loaded successfully');
    } catch (err) {
      console.error('Failed to load protobuf:', err);
      addLog('error', `Failed to load protobuf: ${err.message}`);
    }
  }, []);
  
  // Connect to data WebSocket on mount to get initial state
  useEffect(() => {
    // Only connect if not already connected
    if (!dataWsRef.current || dataWsRef.current.readyState !== WebSocket.OPEN) {
      connectDataWebSocket();
    }
    
    return () => {
      if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.OPEN) {
        dataWsRef.current.close();
      }
    };
  }, []);
  
  // Auto-scroll logs and conversation
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);
  
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory]);
  
  const addLog = (level, message) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      fractionalSecondDigits: 2 
    });
    
    // Special handling for audio chunk messages
    if (message === 'Audio chunk queued for playback') {
      audioChunkCountRef.current++;
      // Only log every 10th audio chunk to reduce spam
      if (audioChunkCountRef.current % 10 === 0) {
        setLogs(prev => [...prev.slice(-100), { 
          timestamp, 
          level, 
          message: `Audio playback in progress (${audioChunkCountRef.current} chunks)` 
        }]);
      }
      return;
    }
    
    // Reset audio chunk counter when assistant finishes speaking
    if (message.startsWith('Assistant:')) {
      audioChunkCountRef.current = 0;
    }
    
    setLogs(prev => {
      // Check if the last log is identical (same message within 100ms)
      if (prev.length > 0) {
        const lastLog = prev[prev.length - 1];
        if (lastLog.message === message && lastLog.level === level) {
          const lastTime = new Date(`1970-01-01T${lastLog.timestamp}Z`).getTime();
          const currentTime = new Date(`1970-01-01T${timestamp}Z`).getTime();
          if (Math.abs(currentTime - lastTime) < 100) {
            return prev; // Skip duplicate log
          }
        }
      }
      return [...prev.slice(-100), { timestamp, level, message }];
    });
  };
  
  const convertFloat32ToS16PCM = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const clampedValue = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = clampedValue < 0 ? clampedValue * 32768 : clampedValue * 32767;
    }
    return int16Array;
  };
  
  const enqueueAudioFromProto = (arrayBuffer) => {
    if (!audioContextRef.current || !frameTypeRef.current) return;
    
    try {
      const parsedFrame = frameTypeRef.current.decode(new Uint8Array(arrayBuffer));
      
      if (parsedFrame?.audio) {
        const currentTime = audioContextRef.current.currentTime;
        const diffTime = currentTime - lastMessageTimeRef.current;
        if (playTimeRef.current === 0 || diffTime > PLAY_TIME_RESET_THRESHOLD_MS) {
          playTimeRef.current = currentTime;
        }
        lastMessageTimeRef.current = currentTime;
        
        const audioVector = Array.from(parsedFrame.audio.audio);
        const audioArray = new Uint8Array(audioVector);
        
        audioContextRef.current.decodeAudioData(audioArray.buffer, (buffer) => {
          const source = audioContextRef.current.createBufferSource();
          source.buffer = buffer;
          source.start(playTimeRef.current);
          source.connect(audioContextRef.current.destination);
          playTimeRef.current += buffer.duration;
          addLog('debug', 'Audio chunk queued for playback');
        }, (error) => {
          console.error('Error decoding audio:', error);
          addLog('error', `Error decoding audio: ${error.message}`);
        });
      }
    } catch (err) {
      console.error('Error processing frame:', err);
      addLog('error', `Error processing frame: ${err.message}`);
    }
  };
  
  const connectDataWebSocket = () => {
    // Prevent duplicate connections
    if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.OPEN) {
      addLog('debug', 'Data WebSocket already connected, skipping reconnection');
      return; // Already connected
    }
    
    // Close any existing connection in connecting state
    if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.CONNECTING) {
      addLog('debug', 'Data WebSocket is connecting, aborting duplicate connection');
      return;
    }
    
    const dataWs = new WebSocket('ws://localhost:8766');
    dataWsRef.current = dataWs;
    
    dataWs.onopen = () => {
      addLog('info', 'Data WebSocket connected');
    };
    
    dataWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'connection') {
          addLog('info', `Connection status: ${data.status}`);
        } else if (data.type === 'transcription') {
          if (data.final) {
            setCurrentInteraction(prev => ({ ...prev, user: data.text }));
            addLog('info', `User: ${data.text}`);
            startTimeRef.current = Date.now();
          } else {
            setCurrentInteraction(prev => ({ ...prev, user: data.text + '...' }));
          }
        } else if (data.type === 'assistant_reply') {
          setCurrentInteraction(prev => ({ ...prev, assistant: data.text }));
          if (data.final && startTimeRef.current) {
            const totalTime = Date.now() - startTimeRef.current;
            setLatencyData(prev => ({ ...prev, total: totalTime }));
            addLog('info', `Assistant: ${data.text}`);
            startTimeRef.current = null;
          }
        } else if (data.type === 'conversation_history') {
          setConversationHistory(data.history);
          addLog('debug', `Received ${data.history.length} conversation items`);
        } else if (data.type === 'system_prompt') {
          setSystemPrompt(data.prompt);
          setTempSystemPrompt(data.prompt);
          addLog('info', 'System prompt received from backend');
        } else if (data.type === 'log') {
          // Handle logs from backend
          if (data.message && !data.message.includes('Sent to frontend')) {
            addLog(data.level, data.message);
          }
        }
      } catch (err) {
        console.error('Error parsing data message:', err);
        addLog('error', `Error parsing message: ${err.message}`);
      }
    };
    
    dataWs.onerror = (error) => {
      console.error('Data WebSocket error:', error);
      addLog('error', 'Data WebSocket error occurred');
    };
    
    dataWs.onclose = () => {
      addLog('info', 'Data WebSocket disconnected');
      dataWsRef.current = null;
    };
    
    return dataWs;
  };
  
  const startRecording = async () => {
    if (!frameTypeRef.current) {
      addLog('error', 'Protobuf not loaded yet');
      return;
    }
    
    try {
      // Initialize audio context
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        latencyHint: 'interactive',
        sampleRate: SAMPLE_RATE
      });
      
      // Clear current interaction and reset audio chunk counter
      setCurrentInteraction({ user: '', assistant: '' });
      playTimeRef.current = 0;
      lastMessageTimeRef.current = 0;
      audioChunkCountRef.current = 0;
      
      // Make sure data WebSocket is connected
      if (!dataWsRef.current || dataWsRef.current.readyState !== WebSocket.OPEN) {
        connectDataWebSocket();
      }
      
      // Connect audio WebSocket
      const ws = new WebSocket('ws://localhost:8765');
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;
      
      addLog('info', 'Connecting to audio WebSocket...');
      
      ws.onopen = async () => {
        addLog('info', 'Audio WebSocket connected');
        setIsConnected(true);
        
        try {
          // Get microphone stream
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              sampleRate: SAMPLE_RATE,
              channelCount: NUM_CHANNELS,
              autoGainControl: true,
              echoCancellation: true,
              noiseSuppression: true,
            }
          });
          
          addLog('info', 'Microphone access granted');
          mediaStreamRef.current = stream;
          
          // Setup audio processing
          scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(512, 1, 1);
          sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
          sourceRef.current.connect(scriptProcessorRef.current);
          scriptProcessorRef.current.connect(audioContextRef.current.destination);
          
          scriptProcessorRef.current.onaudioprocess = (event) => {
            if (!ws || ws.readyState !== WebSocket.OPEN || !frameTypeRef.current) return;
            
            const audioData = event.inputBuffer.getChannelData(0);
            const pcmS16Array = convertFloat32ToS16PCM(audioData);
            const pcmByteArray = new Uint8Array(pcmS16Array.buffer);
            
            const frame = {
              audio: {
                audio: pcmByteArray,
                sampleRate: SAMPLE_RATE,
                numChannels: NUM_CHANNELS
              }
            };
            
            const message = frameTypeRef.current.create(frame);
            const buffer = frameTypeRef.current.encode(message).finish();
            ws.send(buffer);
          };
          
          setIsRecording(true);
          addLog('info', 'Recording started');
          
        } catch (err) {
          console.error('Microphone error:', err);
          addLog('error', `Microphone access denied: ${err.message}`);
          ws.close();
        }
      };
      
      ws.onmessage = (event) => {
        enqueueAudioFromProto(event.data);
      };
      
      ws.onerror = (error) => {
        console.error('Audio WebSocket error:', error);
        addLog('error', 'Audio WebSocket error occurred');
      };
      
      ws.onclose = () => {
        addLog('info', 'Audio WebSocket closed');
        stopRecording();
        setIsConnected(false);
      };
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      addLog('error', `Failed to start recording: ${error.message}`);
    }
  };
  
  const stopRecording = () => {
    // Only log if actually recording
    if (isRecording) {
      addLog('info', 'Stopping recording...');
    }
    
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;
    
    setIsRecording(false);
    if (isRecording) {
      addLog('info', 'Recording stopped');
    }
  };
  
  const updateSystemPrompt = () => {
    if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.OPEN) {
      dataWsRef.current.send(JSON.stringify({
        type: 'update_system_prompt',
        prompt: tempSystemPrompt
      }));
      setSystemPrompt(tempSystemPrompt);
      setIsPromptLocked(true);
      addLog('info', 'System prompt updated and history cleared');
    } else {
      addLog('error', 'Data WebSocket not connected');
    }
  };
  
  const clearHistory = () => {
    if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.OPEN) {
      dataWsRef.current.send(JSON.stringify({
        type: 'clear_history'
      }));
      addLog('info', 'Conversation history cleared');
    } else {
      addLog('error', 'Data WebSocket not connected');
    }
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);
  
  return (
    <div className="app">
      <header className="app-header">
        <h1>Voice Assistant Dev Testing</h1>
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>
      
      <div className="app-content">
        <div className="left-panel">
          <section className="system-prompt-section">
            <div className="section-header">
              <h2>SYSTEM PROMPT</h2>
              <button 
                className={`lock-button ${isPromptLocked ? 'locked' : 'unlocked'}`}
                onClick={() => setIsPromptLocked(!isPromptLocked)}
              >
                {isPromptLocked ? '🔒 Locked' : '🔓 Unlocked'}
              </button>
            </div>
            <textarea
              className="system-prompt-input"
              value={tempSystemPrompt}
              onChange={(e) => setTempSystemPrompt(e.target.value)}
              disabled={isPromptLocked}
              placeholder="Enter system prompt..."
            />
            <button 
              className="update-button"
              onClick={updateSystemPrompt}
              disabled={isPromptLocked || tempSystemPrompt === systemPrompt}
            >
              Update & Clear History
            </button>
          </section>
          
          <section className="recording-section">
            <button
              className={`record-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!frameTypeRef.current}
            >
              {isRecording ? (
                <>
                  <span className="stop-icon">⏹</span>
                  Stop Recording
                </>
              ) : (
                <>
                  <span className="mic-icon">🎤</span>
                  Start Recording
                </>
              )}
            </button>
          </section>
          
          <section className="latencies-section">
            <h2>CURRENT LATENCIES</h2>
            <div className="latency-grid">
              <div className="latency-item">
                <div className="latency-label">LLM</div>
                <div className="latency-value">{latencyData.llm}ms</div>
              </div>
              <div className="latency-item">
                <div className="latency-label">TTS</div>
                <div className="latency-value">{latencyData.tts}ms</div>
              </div>
              <div className="latency-item total">
                <div className="latency-label">Total</div>
                <div className="latency-value">{latencyData.total}ms</div>
              </div>
            </div>
          </section>
          
          <section className="current-interaction-section">
            <h2>Current Interaction</h2>
            <div className="interaction-content">
              <div className="interaction-item">
                <div className="interaction-label">USER</div>
                <div className="interaction-text">
                  {currentInteraction.user || 'Waiting for input...'}
                </div>
              </div>
              <div className="interaction-item">
                <div className="interaction-label">ASSISTANT</div>
                <div className="interaction-text">
                  {currentInteraction.assistant || 'Waiting for response...'}
                </div>
              </div>
            </div>
          </section>
        </div>
        
        <div className="middle-panel">
          <section className="conversation-history-section">
            <div className="section-header">
              <h2>CONVERSATION HISTORY</h2>
              <button className="clear-button" onClick={clearHistory}>
                Clear
              </button>
            </div>
            <div className="conversation-list">
              {conversationHistory.length === 0 ? (
                <div className="history-message" style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
                  No conversation history yet. Start recording to begin.
                </div>
              ) : (
                conversationHistory.map((msg, index) => (
                  <div key={index} className={`history-message ${msg.role}`}>
                    <div className="message-role">{msg.role.toUpperCase()}</div>
                    <div className="message-content">{msg.content}</div>
                    <div className="message-timestamp">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))
              )}
              <div ref={conversationEndRef} />
            </div>
          </section>
        </div>
        
        <div className="right-panel">
          <section className="debug-logs-section">
            <div className="section-header">
              <h2>DEBUG LOGS</h2>
              <button className="clear-button" onClick={() => setLogs([])}>
                Clear
              </button>
            </div>
            <div className="logs-container">
              {logs.map((log, index) => (
                <div key={index} className={`log-entry ${log.level}`}>
                  <span className="log-timestamp">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}