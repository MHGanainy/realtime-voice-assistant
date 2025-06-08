import React, { useState, useRef, useEffect, useCallback } from 'react';
import protobuf from 'protobufjs';
import './App.css';

// Import configuration and services
import { CONFIG, EVENT_STYLES, PIPECAT_PROTO } from './config';
import { TTS_PROVIDERS, DEFAULT_TTS_PROVIDER, getTTSConfig, getTTSProviderOptions } from './modelConfig';
import { LLM_PROVIDERS, DEFAULT_LLM_PROVIDER, getLLMConfig, getLLMProviderOptions } from './modelConfig';
import { STT_PROVIDERS, DEFAULT_STT_PROVIDER, getSTTConfig, getSTTProviderOptions } from './modelConfig';
import { encodeAudioFrame, decodeFrame, playAudioQueue, setupAudioWorklet } from './services/audioService';
import { connectEventWebSocket, handleEventMessage as handleEventMsg, createConversationWebSocket, requestEventStats } from './services/websocketService';

function App() {
  const defaultTTS = getTTSConfig(DEFAULT_TTS_PROVIDER);
  const defaultLLM = getLLMConfig(DEFAULT_LLM_PROVIDER);
  const defaultSTT = getSTTConfig(DEFAULT_STT_PROVIDER);
  
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
    autoScrollLogs: true,
    enableProcessors: true,
    systemPrompt: 'You are a helpful assistant. Keep your responses brief and conversational.',
    showSystemPrompt: false,
    ttsProvider: defaultTTS.provider,
    ttsModel: defaultTTS.model,
    ttsVoice: defaultTTS.voice,
    llmProvider: defaultLLM.provider,
    llmModel: defaultLLM.model,
    sttProvider: defaultSTT.provider,
    sttModel: defaultSTT.model,
    metrics: {
      stt: { ttfb: null, processingTime: null },
      llm: { ttfb: null, processingTime: null },
      tts: { ttfb: null, processingTime: null }
    }
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
    stateUpdatePending: false,
    logBuffer: [],
    logFlushInterval: null
  });

  const updateState = (updates) => setState(prev => ({ ...prev, ...updates }));

  // Destructure state early to use in effects
  const { 
    isRecording, isConnected, isAssistantSpeaking, sessionId, 
    status, conversationHistory, devices, selectedDevice, isMicMuted,
    eventLogs, eventWsConnected, autoScrollLogs,
    enableProcessors, systemPrompt, showSystemPrompt,
    ttsProvider, ttsModel, ttsVoice,
    llmProvider, llmModel,
    sttProvider, sttModel,
    metrics
  } = state;

  const addEventLog = useCallback((eventName, eventData, options = {}) => {
    const logEntry = {
      id: `${Date.now()}-${Math.random()}`,
      eventName,
      data: eventData,
      timestamp: options.timestamp || new Date().toISOString(),
      type: options.type || 'event'
    };
  
    // Add to buffer instead of directly updating state
    refs.current.logBuffer.push(logEntry);
    
    // Check if this is a metrics event and update metrics state immediately
    if (eventName && eventName.includes(':metrics:')) {
      // Get service type directly from the service field
      const serviceType = eventData.service?.toLowerCase();
      console.log('serviceType', serviceType);
      // Only proceed if we have a valid service type
      if (serviceType && ['stt', 'llm', 'tts'].includes(serviceType)) {
        if (eventName.includes(':ttfb') && eventData.ttfb_ms !== undefined) {
          console.log('ttfb', eventData.ttfb_ms);
          // Update state immediately for metrics
          setState(prev => ({
            ...prev,
            metrics: {
              ...prev.metrics,
              [serviceType]: {
                ...prev.metrics[serviceType],
                ttfb: eventData.ttfb_ms
              }
            }
          }));
        } else if (eventName.includes(':processing_time') && eventData.processing_time_ms !== undefined) {
          // Update state immediately for metrics
          setState(prev => ({
            ...prev,
            metrics: {
              ...prev.metrics,
              [serviceType]: {
                ...prev.metrics[serviceType],
                processingTime: eventData.processing_time_ms
              }
            }
          }));
        }
      }
    }
  }, []);

  // Flush log buffer to state periodically
  const flushLogBuffer = useCallback(() => {
    if (refs.current.logBuffer.length > 0) {
      const newLogs = [...refs.current.logBuffer];
      refs.current.logBuffer = [];
      
      setState(prevState => ({
        ...prevState,
        eventLogs: [...prevState.eventLogs, ...newLogs].slice(-500)
      }));
    }
  }, []);

  // Start log flushing when recording starts
  useEffect(() => {
    if (isRecording) {
      // Flush logs every 100ms while recording
      refs.current.logFlushInterval = setInterval(flushLogBuffer, 100);
    } else {
      // Clear interval and flush any remaining logs when recording stops
      if (refs.current.logFlushInterval) {
        clearInterval(refs.current.logFlushInterval);
        refs.current.logFlushInterval = null;
        flushLogBuffer(); // Final flush
      }
    }
    
    return () => {
      if (refs.current.logFlushInterval) {
        clearInterval(refs.current.logFlushInterval);
        refs.current.logFlushInterval = null;
      }
    };
  }, [isRecording, flushLogBuffer]);

  // Initialize protobuf
  useEffect(() => {
    try {
      const root = protobuf.parse(PIPECAT_PROTO).root;
      refs.current.protoRoot = root;
      refs.current.frameType = root.lookupType('pipecat.Frame');
    } catch (error) {
      console.error('Failed to initialize protobuf:', error);
    }
  }, []);

  // Initialize audio devices
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(devices => {
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      updateState({ 
        devices: audioInputs,
        selectedDevice: audioInputs[0]?.deviceId || ''
      });
    });
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (state.autoScrollLogs && refs.current.logEndRef) {
      refs.current.logEndRef.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.eventLogs, state.autoScrollLogs]);

  // Update assistant speaking ref
  useEffect(() => {
    refs.current.isAssistantSpeaking = state.isAssistantSpeaking;
  }, [state.isAssistantSpeaking]);

  // Wrapper for handleEventMessage to bind dependencies
  const handleEventMessage = useCallback((message) => {
    handleEventMsg(message, addEventLog, refs);
  }, [addEventLog]);

  // Handle WebSocket messages
  const handleWebSocketMessage = async (event) => {
    if (!(event.data instanceof ArrayBuffer)) return;

    const frame = decodeFrame(event.data, refs.current.frameType);
    if (!frame) return;

    if (frame.audio) {
      refs.current.audioQueue.push({ 
        bytes: frame.audio.audio, 
        rate: frame.audio.sampleRate 
      });
      
      if (!refs.current.isPlaying) {
        playAudioQueue(refs, updateState);
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
      refs.current.logBuffer = []; // Clear log buffer

      const sessionId = crypto.randomUUID();
      updateState({ 
        sessionId, 
        status: 'Requesting microphone access...', 
        eventLogs: [],
        metrics: {
          stt: { ttfb: null, processingTime: null },
          llm: { ttfb: null, processingTime: null },
          tts: { ttfb: null, processingTime: null }
        }
      });

      // Connect event WebSocket
      connectEventWebSocket(sessionId, refs, updateState, addEventLog, handleEventMessage);

      // Get microphone stream
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
      
      // Setup audio worklet with encoder callback
      const encodeCallback = (audioData) => encodeAudioFrame(audioData, refs.current.frameType);
      await setupAudioWorklet(stream, refs, encodeCallback);

      updateState({ status: 'Connecting to server...' });
      
      // Create conversation WebSocket
      const ws = createConversationWebSocket(
        sessionId,
        { 
          systemPrompt: state.systemPrompt, 
          enableProcessors: state.enableProcessors,
          ttsProvider: state.ttsProvider,
          ttsModel: state.ttsModel,
          ttsVoice: state.ttsVoice,
          llmProvider: state.llmProvider,
          llmModel: state.llmModel,
          sttProvider: state.sttProvider,
          sttModel: state.sttModel
        },
        refs,
        updateState,
        handleWebSocketMessage,
        addEventLog
      );
      
      refs.current.websocket = ws;
    } catch (error) {
      console.error('Failed to start:', error);
      updateState({ status: `Error: ${error.message}` });
    }
  };

  const stopConversation = () => {
    const { websocket, eventWebsocket, mediaStream, audioContext, processor } = refs.current;
    
    // Flush any remaining logs before stopping
    flushLogBuffer();
    
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
    refs.current.logBuffer = []; // Clear log buffer

    updateState({
      isRecording: false,
      isConnected: false,
      isAssistantSpeaking: false,
      isMicMuted: false,
      eventWsConnected: false,
      status: 'Conversation ended'
    });
  };

  const addTestEvent = () => {
    addEventLog('test:event:manual', {
      message: 'This is a test event',
      timestamp: new Date().toISOString(),
      random: Math.random()
    }, { type: 'info' });
  };

  const getEventCategory = (eventName) => {
    if (!eventName || typeof eventName !== 'string') return 'system';
    const parts = eventName.split(':');
    return parts[0] || 'system';
  };

  const clearLogs = () => {
    updateState({ eventLogs: [] });
    addEventLog('system', 'Logs cleared', { type: 'info' });
  };

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

            {/* STT Provider selector */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                color: '#e0e0e0',
                marginBottom: '5px'
              }}>
                Speech Recognition:
              </label>
              <select 
                value={sttProvider} 
                onChange={(e) => {
                  const provider = e.target.value;
                  const wasRecording = isRecording;
                  
                  // Stop current conversation if active
                  if (wasRecording) {
                    stopConversation();
                  }
                  
                  // Update STT settings from config
                  const sttConfig = getSTTConfig(provider);
                  updateState({ 
                    sttProvider: sttConfig.provider,
                    sttModel: sttConfig.model
                  });
                  
                  // Restart conversation if it was active
                  if (wasRecording) {
                    setTimeout(() => {
                      startConversation();
                    }, 500);
                  }
                }}
                disabled={isRecording}
                style={{
                  width: '100%',
                  padding: '8px',
                  background: '#2a2a2a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  fontSize: '0.875rem',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  opacity: isRecording ? 0.6 : 1
                }}
              >
                {getSTTProviderOptions().map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div style={{
                marginTop: '5px',
                fontSize: '0.75rem',
                color: '#666'
              }}>
                {getSTTConfig(sttProvider).description}
              </div>
            </div>

            {/* LLM Provider selector */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                color: '#e0e0e0',
                marginBottom: '5px'
              }}>
                LLM Model:
              </label>
              <select 
                value={llmProvider} 
                onChange={(e) => {
                  const provider = e.target.value;
                  const wasRecording = isRecording;
                  
                  // Stop current conversation if active
                  if (wasRecording) {
                    stopConversation();
                  }
                  
                  // Update LLM settings from config
                  const llmConfig = getLLMConfig(provider);
                  updateState({ 
                    llmProvider: llmConfig.provider,
                    llmModel: llmConfig.model
                  });
                  
                  // Restart conversation if it was active
                  if (wasRecording) {
                    setTimeout(() => {
                      startConversation();
                    }, 500);
                  }
                }}
                disabled={isRecording}
                style={{
                  width: '100%',
                  padding: '8px',
                  background: '#2a2a2a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  fontSize: '0.875rem',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  opacity: isRecording ? 0.6 : 1
                }}
              >
                {getLLMProviderOptions().map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div style={{
                marginTop: '5px',
                fontSize: '0.75rem',
                color: '#666'
              }}>
                {getLLMConfig(llmProvider).description}
              </div>
            </div>

            {/* TTS Provider selector */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                color: '#e0e0e0',
                marginBottom: '5px'
              }}>
                TTS Provider:
              </label>
              <select 
                value={ttsProvider} 
                onChange={(e) => {
                  const provider = e.target.value;
                  const wasRecording = isRecording;
                  
                  // Stop current conversation if active
                  if (wasRecording) {
                    stopConversation();
                  }
                  
                  // Update TTS settings from config
                  const ttsConfig = getTTSConfig(provider);
                  updateState({ 
                    ttsProvider: ttsConfig.provider,
                    ttsModel: ttsConfig.model,
                    ttsVoice: ttsConfig.voice
                  });
                  
                  // Restart conversation if it was active
                  if (wasRecording) {
                    setTimeout(() => {
                      startConversation();
                    }, 500);
                  }
                }}
                disabled={isRecording}
                style={{
                  width: '100%',
                  padding: '8px',
                  background: '#2a2a2a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  fontSize: '0.875rem',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  opacity: isRecording ? 0.6 : 1
                }}
              >
                {getTTSProviderOptions().map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div style={{
                marginTop: '5px',
                fontSize: '0.75rem',
                color: '#666'
              }}>
                {getTTSConfig(ttsProvider).description}
              </div>
            </div>

            {/* System prompt editor */}
            <div style={{ marginBottom: '15px' }}>
              <button
                onClick={() => updateState({ showSystemPrompt: !showSystemPrompt })}
                disabled={isRecording}
                style={{
                  padding: '8px 16px',
                  background: '#2a2a2a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  cursor: isRecording ? 'not-allowed' : 'pointer',
                  fontSize: '0.875rem',
                  marginBottom: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  opacity: isRecording ? 0.6 : 1
                }}
              >
                {showSystemPrompt ? '🔽' : '▶️'} Customize System Prompt
              </button>
              
              {showSystemPrompt && (
                <div style={{
                  background: '#2a2a2a',
                  border: '1px solid #3a3a3a',
                  borderRadius: '8px',
                  padding: '15px',
                  marginBottom: '10px'
                }}>
                  <label style={{
                    display: 'block',
                    fontSize: '0.875rem',
                    color: '#9ca3af',
                    marginBottom: '8px'
                  }}>
                    System Prompt (defines assistant behavior):
                  </label>
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => {
                      const newPrompt = e.target.value;
                      const wasRecording = isRecording;
                      
                      // Stop current conversation if active
                      if (wasRecording) {
                        stopConversation();
                      }
                      
                      // Update the prompt
                      updateState({ systemPrompt: newPrompt });
                      
                      // Restart conversation if it was active
                      if (wasRecording) {
                        setTimeout(() => {
                          startConversation();
                        }, 500);
                      }
                    }}
                    disabled={isRecording}
                    placeholder="Enter system prompt..."
                    style={{
                      width: '100%',
                      minHeight: '100px',
                      maxHeight: '200px',
                      background: '#1a1a1a',
                      border: '1px solid #3a3a3a',
                      borderRadius: '4px',
                      padding: '10px',
                      color: '#e0e0e0',
                      fontSize: '0.875rem',
                      fontFamily: 'Monaco, Menlo, monospace',
                      resize: 'vertical',
                      opacity: isRecording ? 0.6 : 1
                    }}
                  />
                  <div style={{
                    marginTop: '8px',
                    fontSize: '0.75rem',
                    color: '#666'
                  }}>
                    {systemPrompt.length} characters
                  </div>
                </div>
              )}
            </div>

            {/* Processor toggle */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '10px', 
                color: '#e0e0e0',
                fontSize: '0.95rem',
                cursor: isRecording ? 'not-allowed' : 'pointer',
                opacity: isRecording ? 0.6 : 1
              }}>
                <input 
                  type="checkbox" 
                  checked={enableProcessors} 
                  onChange={(e) => updateState({ enableProcessors: e.target.checked })}
                  disabled={isRecording}
                  style={{ cursor: isRecording ? 'not-allowed' : 'pointer' }}
                />
                Enable Conversation Tracking & Metrics
              </label>
              <div style={{ 
                fontSize: '0.8rem', 
                color: enableProcessors ? '#34d399' : '#fbbf24',
                marginLeft: '24px',
                marginTop: '4px'
              }}>
                {enableProcessors 
                  ? '✓ Full tracking: transcriptions saved, metrics collected, events emitted'
                  : '⚡ Low-latency mode: minimal processing, no tracking'}
              </div>
            </div>

            <button 
              className={`main-button ${isRecording ? 'stop' : 'start'}`}
              onClick={isRecording ? stopConversation : startConversation}
            >
              {isRecording ? '⏹️ Stop Conversation' : '▶️ Start Conversation'}
            </button>
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
            {isConnected && (
              <div className={`indicator ${enableProcessors ? 'active' : ''}`} style={{
                background: enableProcessors ? '#3a3a3a' : '#2a2a2a',
                color: enableProcessors ? '#34d399' : '#fbbf24'
              }}>
                {enableProcessors ? '📊 Tracking On' : '⚡ Low Latency'}
              </div>
            )}
          </div>

          {/* Metrics Display */}
          {isConnected && enableProcessors && (
            <div className="metrics-panel" style={{
              marginTop: '20px',
              padding: '15px',
              background: '#2a2a2a',
              border: '1px solid #3a3a3a',
              borderRadius: '8px'
            }}>
              <h3 style={{ margin: '0 0 15px 0', fontSize: '1rem', color: '#e0e0e0' }}>
                ⚡ Performance Metrics
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px' }}>
                {/* STT Metrics */}
                <div style={{
                  padding: '12px',
                  background: '#1a1a1a',
                  borderRadius: '6px',
                  border: '1px solid #3a3a3a'
                }}>
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '0.875rem', color: '#9ca3af' }}>
                    🎤 STT
                  </h4>
                  <div style={{ fontSize: '0.75rem', color: '#e0e0e0' }}>
                    <div style={{ marginBottom: '4px' }}>
                      TTFB: {metrics.stt.ttfb !== null ? `${metrics.stt.ttfb}ms` : '-'}
                    </div>
                    <div>
                      Processing: {metrics.stt.processingTime !== null ? `${metrics.stt.processingTime}ms` : '-'}
                    </div>
                  </div>
                </div>
                
                {/* LLM Metrics */}
                <div style={{
                  padding: '12px',
                  background: '#1a1a1a',
                  borderRadius: '6px',
                  border: '1px solid #3a3a3a'
                }}>
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '0.875rem', color: '#9ca3af' }}>
                    🧠 LLM
                  </h4>
                  <div style={{ fontSize: '0.75rem', color: '#e0e0e0' }}>
                    <div style={{ marginBottom: '4px' }}>
                      TTFB: {metrics.llm.ttfb !== null ? `${metrics.llm.ttfb}ms` : '-'}
                    </div>
                    <div>
                      Processing: {metrics.llm.processingTime !== null ? `${metrics.llm.processingTime}ms` : '-'}
                    </div>
                  </div>
                </div>
                
                {/* TTS Metrics */}
                <div style={{
                  padding: '12px',
                  background: '#1a1a1a',
                  borderRadius: '6px',
                  border: '1px solid #3a3a3a'
                }}>
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '0.875rem', color: '#9ca3af' }}>
                    🔊 TTS
                  </h4>
                  <div style={{ fontSize: '0.75rem', color: '#e0e0e0' }}>
                    <div style={{ marginBottom: '4px' }}>
                      TTFB: {metrics.tts.ttfb !== null ? `${metrics.tts.ttfb}ms` : '-'}
                    </div>
                    <div>
                      Processing: {metrics.tts.processingTime !== null ? `${metrics.tts.processingTime}ms` : '-'}
                    </div>
                  </div>
                </div>
              </div>
              <div style={{
                marginTop: '10px',
                fontSize: '0.7rem',
                color: '#666',
                textAlign: 'center'
              }}>
                TTFB: Time To First Byte | Processing: Total processing time
              </div>
            </div>
          )}

          <div className="tips" style={{ marginTop: '20px' }}>
            <h3>💡 Tips:</h3>
            <ul>
              <li>Speak naturally, the assistant will respond</li>
              <li>Say 'goodbye' to end the conversation</li>
              <li>Check the Event Log panel to see real-time events</li>
              <li>Use Debug Mode to see detailed console logs</li>
              <li>Disable processors for lower latency (no tracking)</li>
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
            {eventLogs.length === 0 ? (
              <div className="log-empty">
                No events to display
                {!enableProcessors && isConnected && (
                  <>
                    <br/>
                    <small style={{ color: '#fbbf24', fontSize: '0.9em' }}>
                      Note: Processors are disabled - fewer events will be logged
                    </small>
                  </>
                )}
              </div>
            ) : (
              eventLogs.map((log) => {
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