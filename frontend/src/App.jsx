import React, { useState, useRef, useEffect, useCallback } from 'react';
import protobuf from 'protobufjs';
import './App.css';

const CONFIG = {
  MIC_SAMPLE_RATE: 16000,
  CHANNELS: 1,
  CHUNK_SIZE: 320,
  WS_BASE_URL: 'ws://localhost:8000',
  API_BASE_URL: 'http://localhost:8000',
  EXIT_WORDS: ['goodbye', 'bye', 'exit', 'quit']
};

const PIPECAT_PROTO = `
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

message MessageFrame {
    string data = 1;
}

message Frame {
  oneof frame {
    TextFrame text = 1;
    AudioRawFrame audio = 2;
    TranscriptionFrame transcription = 3;
    MessageFrame message = 4;
  }
}
`;

const EVENT_STYLES = {
  'connection': { icon: '🔌', color: '#60a5fa' },
  'conversation': { icon: '💬', color: '#34d399' },
  'transcription': { icon: '📝', color: '#a78bfa' },
  'audio': { icon: '🔊', color: '#fbbf24' },
  'turn': { icon: '🔄', color: '#f472b6' },
  'metrics': { icon: '📊', color: '#94a3b8' },
  'error': { icon: '❌', color: '#ef4444' },
  'pipeline': { icon: '⚙️', color: '#06b6d4' },
  'global': { icon: '🌐', color: '#6366f1' },
  'system': { icon: '💻', color: '#8b5cf6' },
  'test': { icon: '🧪', color: '#ec4899' }
};

function App() {
  const [state, setState] = useState({
    isRecording: false,
    isConnected: false,
    isAssistantSpeaking: false,
    sessionId: '',
    status: 'Ready to start',
    conversationHistory: [],
    devices: [],
    selectedDevice: '',
    isMicMuted: false,
    eventLogs: [],
    eventWsConnected: false,
    eventFilter: 'all',
    autoScrollLogs: true,
    debugMode: false
  });

  const refs = useRef({
    audioContext: null,
    mediaStream: null,
    processor: null,
    websocket: null,
    eventWebsocket: null,
    audioQueue: [],
    isPlaying: false,
    protoRoot: null,
    frameType: null,
    isAssistantSpeaking: false,
    logEndRef: null,
    eventPingInterval: null,
    nextStartTime: 0,
    stateUpdatePending: false
  });

  const updateState = (updates) => setState(prev => ({ ...prev, ...updates }));

  const addEventLog = useCallback((eventName, eventData, options = {}) => {
    const logEntry = {
      id: `${Date.now()}-${Math.random()}`,
      eventName,
      data: eventData,
      timestamp: options.timestamp || new Date().toISOString(),
      type: options.type || 'event'
    };
  
    setState(prevState => ({
      ...prevState,
      eventLogs: [...prevState.eventLogs.slice(-499), logEntry]
    }));
  }, []);

  useEffect(() => {
    try {
      const root = protobuf.parse(PIPECAT_PROTO).root;
      refs.current.protoRoot = root;
      refs.current.frameType = root.lookupType('pipecat.Frame');
    } catch (error) {
      console.error('Failed to initialize protobuf:', error);
    }
  }, []);

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(devices => {
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      updateState({ 
        devices: audioInputs,
        selectedDevice: audioInputs[0]?.deviceId || ''
      });
    });
  }, []);

  useEffect(() => {
    if (state.autoScrollLogs && refs.current.logEndRef) {
      refs.current.logEndRef.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.eventLogs, state.autoScrollLogs]);

  const startEventPing = () => {
    const pingInterval = setInterval(() => {
      if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
        refs.current.eventWebsocket.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      } else {
        clearInterval(pingInterval);
      }
    }, 30000);
    
    refs.current.eventPingInterval = pingInterval;
  };

  const connectEventWebSocket = (sessionId) => {
    if (!sessionId) return;

    const eventWs = new WebSocket(`${CONFIG.WS_BASE_URL}/ws/events?session_id=${sessionId}`);
    
    eventWs.onopen = () => {
      updateState({ eventWsConnected: true });
      addEventLog('system', 'Event stream connected', { type: 'info' });
      startEventPing();
    };

    eventWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEventMessage(data);
      } catch (error) {
        addEventLog('system', 'Failed to parse event message', { 
          type: 'error',
          data: { error: error.message }
        });
      }
    };

    eventWs.onerror = (error) => {
      addEventLog('system', 'Event stream error', { type: 'error' });
    };

    eventWs.onclose = (event) => {
      updateState({ eventWsConnected: false });
      addEventLog('system', 'Event stream disconnected', { 
        type: 'warning',
        data: { code: event.code, reason: event.reason }
      });
      
      if (refs.current.eventPingInterval) {
        clearInterval(refs.current.eventPingInterval);
        refs.current.eventPingInterval = null;
      }
      
      refs.current.eventWebsocket = null;
    };

    refs.current.eventWebsocket = eventWs;
  };

  const handleEventMessage = useCallback((message) => {
    const { type, event: eventName, data, timestamp } = message;
  
    switch (type) {
      case 'event':
        if (eventName) {
          addEventLog(eventName, data || {}, { timestamp });
        }
        break;
  
      case 'welcome':
        addEventLog('system:welcome', data || {}, { type: 'success' });
        
        if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
          refs.current.eventWebsocket.send(JSON.stringify({
            type: 'subscribe',
            subscription: {
              patterns: ["*"],
              include_historical: true
            }
          }));
        }
        break;
  
      case 'historical_start':
        addEventLog('system:historical:start', { count: data?.count || 0 }, { type: 'info' });
        break;
  
      case 'historical_end':
        addEventLog('system:historical:end', { count: data?.count || 0 }, { type: 'success' });
        break;
  
      case 'subscription_updated':
        addEventLog('system:subscription:updated', data || {}, { type: 'info' });
        break;
  
      case 'stats':
        addEventLog('system:stats', data || {}, { type: 'info' });
        break;

      case 'heartbeat':
      case 'pong':
        break;
  
      default:
        addEventLog('system:unknown', { type, ...message }, { type: 'warning' });
    }
  }, [addEventLog]);

  const addTestEvent = () => {
    addEventLog('test:event:manual', {
      message: 'This is a test event',
      timestamp: new Date().toISOString(),
      random: Math.random()
    }, { type: 'info' });
  };

  const requestEventStats = () => {
    if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
      refs.current.eventWebsocket.send(JSON.stringify({
        type: 'get_stats'
      }));
    }
  };

  const getEventCategory = (eventName) => {
    if (!eventName || typeof eventName !== 'string') return 'system';
    const parts = eventName.split(':');
    return parts[0] || 'system';
  };

  const getFilteredLogs = () => {
    if (state.eventFilter === 'all') return state.eventLogs;
    return state.eventLogs.filter(log => {
      const category = getEventCategory(log.eventName);
      return category === state.eventFilter;
    });
  };

  const clearLogs = () => {
    updateState({ eventLogs: [] });
    addEventLog('system', 'Logs cleared', { type: 'info' });
  };

  useEffect(() => {
    refs.current.isAssistantSpeaking = state.isAssistantSpeaking;
  }, [state.isAssistantSpeaking]);

  const encodeAudioFrame = (audioData) => {
    const { frameType } = refs.current;
    if (!frameType) return null;

    try {
      const message = frameType.create({
        audio: {
          audio: audioData,
          sampleRate: CONFIG.MIC_SAMPLE_RATE,
          numChannels: CONFIG.CHANNELS,
        }
      });
      return frameType.encode(message).finish();
    } catch (error) {
      console.error("Encode error:", error);
      return null;
    }
  };

  const decodeFrame = (data) => {
    const { frameType } = refs.current;
    if (!frameType) return null;

    try {
      const decoded = frameType.decode(new Uint8Array(data));
      return frameType.toObject(decoded, {
        bytes: Uint8Array,
        longs: Number,
        defaults: true,
      });
    } catch (error) {
      console.error("Decode error:", error);
      return null;
    }
  };

  const playAudioQueue = async () => {
    const { audioQueue, audioContext } = refs.current;
    
    // Check if already playing
    if (refs.current.isPlaying || !audioQueue.length || !audioContext) return;

    refs.current.isPlaying = true;
    
    // Only update state once at the beginning
    if (!refs.current.isAssistantSpeaking) {
      refs.current.isAssistantSpeaking = true;
      updateState({ isAssistantSpeaking: true, isMicMuted: true });
    }

    // Initialize next start time if needed
    if (refs.current.nextStartTime < audioContext.currentTime) {
      refs.current.nextStartTime = audioContext.currentTime + 0.05;
    }

    while (audioQueue.length > 0 && refs.current.audioContext) {
      const { bytes: audioData, rate } = audioQueue.shift();

      try {
        if (refs.current.audioContext.state === 'closed') break;

        const aligned = audioData.byteOffset & 1 ? new Uint8Array(audioData) : audioData;
        const int16 = new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength >> 1);
        const float32 = new Float32Array(int16.length);
        
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768;
        }

        const buffer = refs.current.audioContext.createBuffer(1, float32.length, rate);
        buffer.getChannelData(0).set(float32);

        const source = refs.current.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(refs.current.audioContext.destination);

        // Schedule playback for seamless audio
        source.start(refs.current.nextStartTime);
        
        // Calculate next start time
        const duration = buffer.duration;
        refs.current.nextStartTime += duration;

        // Wait for this chunk to finish
        await new Promise(resolve => {
          source.onended = resolve;
        });

      } catch (error) {
        console.error("Playback error:", error);
        break;
      }
    }

    refs.current.isPlaying = false;
    refs.current.isAssistantSpeaking = false;
    refs.current.nextStartTime = 0;
    updateState({ isAssistantSpeaking: false, isMicMuted: false, status: "Ready for next input" });
  };

  const setupAudioWorklet = async (stream) => {
    const audioContext = new AudioContext({ sampleRate: CONFIG.MIC_SAMPLE_RATE });
    refs.current.audioContext = audioContext;

    // Ensure audio context is running
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    const processorCode = `
      class AudioProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.buffer = [];
        }

        process(inputs) {
          const input = inputs[0]?.[0];
          if (!input) return true;
          
          for (const sample of input) {
            this.buffer.push(sample);
            
            if (this.buffer.length >= ${CONFIG.CHUNK_SIZE}) {
              const int16 = new Int16Array(${CONFIG.CHUNK_SIZE});
              for (let i = 0; i < ${CONFIG.CHUNK_SIZE}; i++) {
                const s = Math.max(-1, Math.min(1, this.buffer[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
              }
              
              this.port.postMessage({ type: 'audio', data: int16.buffer });
              this.buffer = [];
            }
          }
          return true;
        }
      }
      registerProcessor('audio-processor', AudioProcessor);
    `;

    const blob = new Blob([processorCode], { type: 'application/javascript' });
    await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));

    const source = audioContext.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioContext, 'audio-processor');
    
    processor.port.onmessage = (event) => {
      if (event.data.type === 'audio' && 
          !refs.current.isAssistantSpeaking && 
          refs.current.websocket?.readyState === WebSocket.OPEN) {
        const encoded = encodeAudioFrame(new Uint8Array(event.data.data));
        if (encoded) {
          refs.current.websocket.send(encoded);
        }
      }
    };

    source.connect(processor);
    // DON'T connect processor to destination - this causes feedback!
    
    refs.current.processor = processor;
  };

  const handleWebSocketMessage = async (event) => {
    if (!(event.data instanceof ArrayBuffer)) return;

    const frame = decodeFrame(event.data);
    if (!frame) return;

    if (frame.audio) {
      // Don't update state every time - just queue the audio
      refs.current.audioQueue.push({ 
        bytes: frame.audio.audio, 
        rate: frame.audio.sampleRate 
      });
      
      // Only start playback if not already playing
      if (!refs.current.isPlaying) {
        playAudioQueue();
      }
    } else if (frame.text) {
      updateState(prev => ({
        conversationHistory: [...prev.conversationHistory, { 
          speaker: 'assistant', 
          text: frame.text.text 
        }]
      }));
    } else if (frame.transcription) {
      const text = frame.transcription.text;
      updateState(prev => ({
        conversationHistory: [...prev.conversationHistory, { 
          speaker: 'user', 
          text 
        }]
      }));
      
      if (CONFIG.EXIT_WORDS.some(word => text.toLowerCase().includes(word))) {
        updateState({ status: 'Goodbye! Thanks for chatting!' });
        setTimeout(stopConversation, 2000);
      }
    }
  };

  const startConversation = async () => {
    try {
      refs.current.audioQueue = [];
      refs.current.isPlaying = false;
      refs.current.isAssistantSpeaking = false;
      refs.current.nextStartTime = 0;

      const sessionId = crypto.randomUUID();
      updateState({ sessionId, status: 'Requesting microphone access...', eventLogs: [] });

      connectEventWebSocket(sessionId);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: state.selectedDevice,
          channelCount: CONFIG.CHANNELS,
          sampleRate: CONFIG.MIC_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      refs.current.mediaStream = stream;

      updateState({ status: 'Setting up audio processing...' });
      await setupAudioWorklet(stream);

      updateState({ status: 'Connecting to server...' });
      
      const params = new URLSearchParams({
        session_id: sessionId,
        stt_provider: 'deepgram',
        llm_provider: 'openai',
        llm_model: 'gpt-3.5-turbo',
        tts_provider: 'deepinfra',
        system_prompt: 'You are a helpful assistant. Keep your responses brief and conversational.',
        enable_interruptions: 'false',
        vad_enabled: 'true'
      });
      
      const ws = new WebSocket(`${CONFIG.WS_BASE_URL}/ws/conversation?${params}`);
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => updateState({ 
        isConnected: true, 
        isRecording: true, 
        status: 'Connected! Speak naturally...' 
      });
      
      ws.onmessage = handleWebSocketMessage;
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateState({ status: 'Connection error' });
      };
      ws.onclose = (event) => {
        updateState({ 
          isConnected: false, 
          isRecording: false, 
          status: `Disconnected: ${event.reason || 'Connection closed'}` 
        });
      };
      
      refs.current.websocket = ws;
    } catch (error) {
      console.error('Failed to start:', error);
      updateState({ status: `Error: ${error.message}` });
    }
  };

  const stopConversation = () => {
    const { websocket, eventWebsocket, mediaStream, audioContext, processor } = refs.current;
    
    if (websocket) {
      websocket.close();
      refs.current.websocket = null;
    }
    
    if (eventWebsocket) {
      eventWebsocket.close();
      refs.current.eventWebsocket = null;
    }
    
    if (refs.current.eventPingInterval) {
      clearInterval(refs.current.eventPingInterval);
      refs.current.eventPingInterval = null;
    }
    
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      refs.current.mediaStream = null;
    }
    
    if (processor) {
      processor.disconnect();
      refs.current.processor = null;
    }
    
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
      refs.current.audioContext = null;
    }
    
    refs.current.audioQueue = [];
    refs.current.isPlaying = false;
    refs.current.isAssistantSpeaking = false;
    refs.current.nextStartTime = 0;

    updateState({
      isRecording: false,
      isConnected: false,
      isAssistantSpeaking: false,
      isMicMuted: false,
      eventWsConnected: false,
      status: 'Conversation ended'
    });
  };

  const { 
    isRecording, isConnected, isAssistantSpeaking, sessionId, 
    status, conversationHistory, devices, selectedDevice, isMicMuted,
    eventLogs, eventWsConnected, eventFilter, autoScrollLogs, debugMode
  } = state;

  return (
    <div className="app">
      <div className="main-container">
        <div className="conversation-panel">
          <h1>🎙️ Interactive Voice Assistant</h1>
          
          <div className="status-bar">
            <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
              {status}
            </div>
            {sessionId && <div className="session-id">Session: {sessionId.slice(0, 8)}...</div>}
          </div>

          <div className="controls">
            <div className="device-selector">
              <label>Audio Device:</label>
              <select 
                value={selectedDevice} 
                onChange={(e) => updateState({ selectedDevice: e.target.value })}
                disabled={isRecording}
              >
                {devices.map(device => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Microphone ${device.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            </div>

            <button 
              className={`main-button ${isRecording ? 'stop' : 'start'}`}
              onClick={isRecording ? stopConversation : startConversation}
            >
              {isRecording ? '⏹️ Stop Conversation' : '▶️ Start Conversation'}
            </button>

            {debugMode && (
              <div className="debug-controls" style={{ marginTop: '10px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button 
                  onClick={addTestEvent}
                  style={{
                    padding: '8px 16px',
                    background: '#9333ea',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    cursor: 'pointer',
                    fontSize: '0.875rem'
                  }}
                >
                  ➕ Add Test Event
                </button>
                
                <button 
                  onClick={requestEventStats}
                  style={{
                    padding: '8px 16px',
                    background: '#059669',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    cursor: 'pointer',
                    fontSize: '0.875rem'
                  }}
                >
                  📊 Request Stats
                </button>
              </div>
            )}
            
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#e0e0e0', marginTop: '10px' }}>
              <input 
                type="checkbox" 
                checked={debugMode} 
                onChange={(e) => updateState({ debugMode: e.target.checked })}
              />
              Debug Mode
            </label>
          </div>

          <div className="indicators">
            <div className={`indicator ${isRecording && !isMicMuted ? 'active' : ''}`}>
              🎤 {isRecording ? (isMicMuted ? 'Mic Muted' : 'Recording') : 'Not Recording'}
            </div>
            <div className={`indicator ${isAssistantSpeaking ? 'active' : ''}`}>
              🤖 {isAssistantSpeaking ? 'Assistant Speaking' : 'Assistant Idle'}
            </div>
            <div className={`indicator ${eventWsConnected ? 'active' : ''}`}>
              📡 {eventWsConnected ? 'Events Connected' : 'Events Disconnected'}
            </div>
          </div>

          <div className="tips" style={{ marginTop: '20px' }}>
            <h3>💡 Tips:</h3>
            <ul>
              <li>Speak naturally, the assistant will respond</li>
              <li>Say 'goodbye' to end the conversation</li>
              <li>Check the Event Log panel to see real-time events</li>
              <li>Use Debug Mode to see detailed console logs</li>
            </ul>
          </div>

          {conversationHistory.length > 0 && (
            <div className="history">
              <h3>Conversation History</h3>
              {conversationHistory.map((entry, index) => (
                <div key={index} className={`message ${entry.speaker}`}>
                  <span className="speaker">{entry.speaker === 'user' ? '👤' : '🤖'}</span>
                  <span className="text">{entry.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="event-log-panel">
          <div className="log-header">
            <h3>📊 Event Log ({eventLogs.length} events)</h3>
            <div className="log-controls">
              <select 
                className="filter-select"
                value={eventFilter} 
                onChange={(e) => updateState({ eventFilter: e.target.value })}
              >
                <option value="all">All Events</option>
                <option value="connection">Connection</option>
                <option value="conversation">Conversation</option>
                <option value="transcription">Transcription</option>
                <option value="audio">Audio</option>
                <option value="turn">Turn</option>
                <option value="metrics">Metrics</option>
                <option value="error">Errors</option>
                <option value="pipeline">Pipeline</option>
                <option value="global">Global</option>
                <option value="system">System</option>
                <option value="test">Test</option>
              </select>
              <button 
                className={`toggle-button ${autoScrollLogs ? 'active' : ''}`}
                onClick={() => updateState({ autoScrollLogs: !autoScrollLogs })}
                title="Auto-scroll"
              >
                📜
              </button>
              <button 
                className="clear-button"
                onClick={clearLogs}
                title="Clear logs"
              >
                🗑️
              </button>
            </div>
          </div>

          <div className="log-content">
            {getFilteredLogs().length === 0 ? (
              <div className="log-empty">
                No events to display
                {debugMode && (
                  <>
                    <br/>
                    <small style={{ color: '#666', fontSize: '0.8em' }}>
                      Click "Add Test Event" to verify the log panel is working
                    </small>
                  </>
                )}
              </div>
            ) : (
              getFilteredLogs().map((log) => {
                const category = getEventCategory(log.eventName);
                const style = EVENT_STYLES[category] || { icon: '📌', color: '#94a3b8' };
                const time = new Date(log.timestamp).toLocaleTimeString();
                
                return (
                  <div key={log.id} className={`log-entry ${log.type}`}>
                    <div className="log-time">{time}</div>
                    <div className="log-icon" style={{ color: style.color }}>
                      {style.icon}
                    </div>
                    <div className="log-details">
                      <div className="log-event-name" style={{ color: style.color }}>
                        {log.eventName}
                      </div>
                      {log.data && typeof log.data === 'object' && Object.keys(log.data).length > 0 && (
                        <div className="log-data">
                          {Object.entries(log.data).map(([key, value]) => (
                            <div key={key} className="log-data-item">
                              <span className="log-data-key">{key}:</span>
                              <span className="log-data-value">
                                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={el => refs.current.logEndRef = el} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;