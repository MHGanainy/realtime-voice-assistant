import { useState, useRef, useEffect } from 'react';
import protobuf from 'protobufjs';
import './App.css';
import AudioDebugToolkit from './AudioDebugToolkit';

// Session management
const getSessionId = () => {
  let sessionId = sessionStorage.getItem('voice-assistant-session-id');
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem('voice-assistant-session-id', sessionId);
  }
  return sessionId;
};

// Model definitions
const AVAILABLE_MODELS = [
  // OpenAI Models
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', service: 'openai', category: 'OpenAI' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini', service: 'openai', category: 'OpenAI' },
  
  // Meta Llama Models
  { id: 'meta-llama/Meta-Llama-3.3-70B-Instruct', name: 'Llama 3.3 70B', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Meta-Llama-3.1-405B-Instruct', name: 'Llama 3.1 405B', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Meta-Llama-3.1-70B-Instruct', name: 'Llama 3.1 70B', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Meta-Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Llama-3.2-11B-Vision-Instruct', name: 'Llama 3.2 11B Vision', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Llama-3.2-3B-Instruct', name: 'Llama 3.2 3B', service: 'deepinfra', category: 'Meta Llama' },
  { id: 'meta-llama/Llama-3.2-1B-Instruct', name: 'Llama 3.2 1B', service: 'deepinfra', category: 'Meta Llama' },
  
  // DeepSeek Models
  { id: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek V3', service: 'deepinfra', category: 'DeepSeek' },
  { id: 'deepseek-ai/DeepSeek-R1', name: 'DeepSeek R1', service: 'deepinfra', category: 'DeepSeek' },
  { id: 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', name: 'DeepSeek R1 Distill 70B', service: 'deepinfra', category: 'DeepSeek' },
  
  // Qwen Models
  { id: 'Qwen/QwQ-32B', name: 'QwQ 32B', service: 'deepinfra', category: 'Qwen' },
  { id: 'Qwen/Qwen2.5-72B-Instruct', name: 'Qwen 2.5 72B', service: 'deepinfra', category: 'Qwen' },
  { id: 'Qwen/Qwen2.5-7B-Instruct', name: 'Qwen 2.5 7B', service: 'deepinfra', category: 'Qwen' },
  { id: 'Qwen/Qwen2.5-Coder-32B-Instruct', name: 'Qwen 2.5 Coder 32B', service: 'deepinfra', category: 'Qwen' },
  
  // Google Models
  { id: 'google/gemma-3-27b-it', name: 'Gemma 3 27B', service: 'deepinfra', category: 'Google' },
  { id: 'google/gemma-3-12b-it', name: 'Gemma 3 12B', service: 'deepinfra', category: 'Google' },
  { id: 'google/gemma-3-4b-it', name: 'Gemma 3 4B', service: 'deepinfra', category: 'Google' },
  { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash', service: 'deepinfra', category: 'Google' },
  
  // Microsoft Models
  { id: 'microsoft/phi-4', name: 'Phi 4', service: 'deepinfra', category: 'Microsoft' },
  { id: 'microsoft/phi-4-reasoning-plus', name: 'Phi 4 Reasoning+', service: 'deepinfra', category: 'Microsoft' },
  { id: 'microsoft/WizardLM-2-8x22B', name: 'WizardLM 2 8x22B', service: 'deepinfra', category: 'Microsoft' },
  
  // Mistral Models
  { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B', service: 'deepinfra', category: 'Mistral' },
  { id: 'mistralai/Mixtral-8x22B-Instruct-v0.1', name: 'Mixtral 8x22B', service: 'deepinfra', category: 'Mistral' },
  { id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral 7B v0.3', service: 'deepinfra', category: 'Mistral' },
  { id: 'mistralai/Mistral-Nemo-Instruct-2407', name: 'Mistral Nemo 12B', service: 'deepinfra', category: 'Mistral' }
];

export default function App() {
  // ---------------------------------------------------------------------------
  // State ---------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const [sessionId] = useState(getSessionId());
  const [isConnected, setIsConnected] = useState(false);
  const [isDataConnected, setIsDataConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don't include special characters in your answers."
  );
  const [tempSystemPrompt, setTempSystemPrompt] = useState(systemPrompt);
  const [isPromptLocked, setIsPromptLocked] = useState(true);

  const [conversationHistory, setConversationHistory] = useState([]);
  const [logs, setLogs] = useState([]);
  const [currentInteraction, setCurrentInteraction] = useState({ user: '', assistant: '' });
  const [latencyData, setLatencyData] = useState({ stt: 0, llm: 0, tts: 0, total: 0 });

  const [sttService, setSttService] = useState('openai');
  const [llmModel, setLlmModel] = useState('gpt-3.5-turbo');
  const [llmService, setLlmService] = useState('openai');
  const [ttsService, setTtsService] = useState('elevenlabs');
  const [notification, setNotification] = useState('');
  const [showDebugToolkit, setShowDebugToolkit] = useState(false);

  // ---------------------------------------------------------------------------
  // Refs ----------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const wsRef = useRef(null); // audio WS
  const dataWsRef = useRef(null); // data WS
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
  const reconnectTimeoutRef = useRef(null);

  // ---------------------------------------------------------------------------
  // Constants -----------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const SAMPLE_RATE = 16000;
  const NUM_CHANNELS = 1;
  const PLAY_TIME_RESET_THRESHOLD_MS = 1.0;
  
  // WebSocket URLs - use environment variable or default to localhost
  const getWebSocketBaseUrl = () => {
    const envUrl = import.meta.env.VITE_WS_URL;
    if (envUrl) {
      // Remove any trailing /ws or / from the environment URL
      return envUrl.replace(/\/ws\/?$/, '').replace(/\/$/, '');
    }
    // Default to localhost for development
    return 'ws://localhost:8000';
  };

  const WS_BASE_URL = getWebSocketBaseUrl();
  const DATA_WS_URL = `${WS_BASE_URL}/ws/data?session=${sessionId}`;
  const AUDIO_WS_URL = `${WS_BASE_URL}/ws/audio?session=${sessionId}`;
  
  // Log URLs for debugging
  console.log('WebSocket Configuration:', {
    BASE_URL: WS_BASE_URL,
    DATA_URL: DATA_WS_URL,
    AUDIO_URL: AUDIO_WS_URL,
    ENV_URL: import.meta.env.VITE_WS_URL
  });

  // ---------------------------------------------------------------------------
  // Effect: Load protobuf schema once -----------------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (frameTypeRef.current) return; // already loaded (StrictMode double‑mount)

    const protoDefinition = `
      syntax = "proto3";
      package pipecat;

      message TextFrame {
        uint64 id   = 1;
        string name = 2;
        string text = 3;
      }

      message AudioRawFrame {
        uint64 id          = 1;
        string name        = 2;
        bytes  audio       = 3;
        uint32 sample_rate = 4;
        uint32 num_channels= 5;
        optional uint64 pts = 6;
      }

      message TranscriptionFrame {
        uint64 id        = 1;
        string name      = 2;
        string text      = 3;
        string user_id   = 4;
        string timestamp = 5;
      }

      message Frame {
        oneof frame {
          TextFrame          text          = 1;
          AudioRawFrame      audio         = 2;
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

  // ---------------------------------------------------------------------------
  // Effect: connect data WS on mount ------------------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => {
    addLog('info', `Session ID: ${sessionId}`);
    connectDataWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      dataWsRef.current?.close();
    };
  }, [sessionId]);

  // ---------------------------------------------------------------------------
  // Effect: auto‑scroll logs + conversation -----------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory]);

  // ---------------------------------------------------------------------------
  // Effect: clear notification after 5 s --------------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!notification) return;
    const t = setTimeout(() => setNotification(''), 5000);
    return () => clearTimeout(t);
  }, [notification]);

  // ---------------------------------------------------------------------------
  // Helper: logging -----------------------------------------------------------
  // ---------------------------------------------------------------------------
  const addLog = (level, message) => {
    const timestamp = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 2,
    });

    // group audio chunk messages to avoid spam
    if (message === 'Audio chunk queued for playback') {
      audioChunkCountRef.current += 1;
      if (audioChunkCountRef.current % 10 === 0) {
        setLogs((p) => [
          ...p.slice(-100),
          {
            timestamp,
            level,
            message: `Audio playback in progress (${audioChunkCountRef.current} chunks)`,
          },
        ]);
      }
      return;
    }

    // reset chunk counter when assistant finishes
    if (message.startsWith('Assistant:')) audioChunkCountRef.current = 0;

    setLogs((p) => {
      // deduplicate identical messages within 100 ms
      if (p.length) {
        const last = p[p.length - 1];
        if (last.message === message && last.level === level) {
          const lastTime = new Date(`1970-01-01T${last.timestamp}Z`).getTime();
          const thisTime = new Date(`1970-01-01T${timestamp}Z`).getTime();
          if (Math.abs(thisTime - lastTime) < 100) return p;
        }
      }
      return [...p.slice(-100), { timestamp, level, message }];
    });
  };

  // ---------------------------------------------------------------------------
  // Helper: PCM conversion ----------------------------------------------------
  // ---------------------------------------------------------------------------
  const convertFloat32ToS16PCM = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const v = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = v < 0 ? v * 32768 : v * 32767;
    }
    return int16Array;
  };

  // ---------------------------------------------------------------------------
  // Helper: enqueue audio frames ---------------------------------------------
  // ---------------------------------------------------------------------------
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

        audioContextRef.current.decodeAudioData(
          audioArray.buffer,
          (buffer) => {
            const source = audioContextRef.current.createBufferSource();
            source.buffer = buffer;
            source.start(playTimeRef.current);
            source.connect(audioContextRef.current.destination);
            playTimeRef.current += buffer.duration;
            addLog('debug', 'Audio chunk queued for playback');
          },
          (error) => {
            console.error('Error decoding audio:', error);
            addLog('error', `Error decoding audio: ${error.message}`);
          },
        );
      }
    } catch (err) {
      console.error('Error processing frame:', err);
      addLog('error', `Error processing frame: ${err.message}`);
    }
  };

  // ---------------------------------------------------------------------------
  // WebSocket: DATA channel ---------------------------------------------------
  // ---------------------------------------------------------------------------
  const connectDataWebSocket = () => {
    if (dataWsRef.current?.readyState === WebSocket.OPEN) return;
    if (dataWsRef.current?.readyState === WebSocket.CONNECTING) return;

    try {
      addLog('info', `Connecting to Data WebSocket: ${DATA_WS_URL}`);
      const ws = new WebSocket(DATA_WS_URL);
      dataWsRef.current = ws;

      ws.onopen = () => {
        addLog('info', 'Data WebSocket connected to FastAPI');
        setIsDataConnected(true);
        setNotification('');
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onerror = (e) => {
        console.error('Data WS error', e);
        addLog('error', `Data WebSocket connection error - ${e.type || 'Unknown error'}`);
        setNotification('Cannot connect to backend. Please check if the backend URL is correct.');
      };

      ws.onclose = (ev) => {
        addLog('warning', `Data WebSocket closed (code ${ev.code}, reason: ${ev.reason || 'No reason provided'})`);
        setIsDataConnected(false);
        dataWsRef.current = null;
        
        // Only reconnect if it wasn't a normal closure
        if (ev.code !== 1000 && !reconnectTimeoutRef.current) {
          const delay = 3000;
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            addLog('info', 'Attempting to reconnect data WebSocket…');
            connectDataWebSocket();
          }, delay);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'connection':
              addLog('info', `Connection status: ${data.status}`);
              if (data.session_id && data.session_id !== sessionId) {
                // Server assigned a different session ID
                sessionStorage.setItem('voice-assistant-session-id', data.session_id);
                window.location.reload(); // Reload to use new session ID
              }
              break;
            case 'transcription':
              if (data.final) {
                setCurrentInteraction((p) => ({ ...p, user: data.text }));
                addLog('info', `User: ${data.text}`);
                startTimeRef.current = Date.now();
              } else {
                setCurrentInteraction((p) => ({ ...p, user: `${data.text}…` }));
              }
              break;
            case 'assistant_reply':
              setCurrentInteraction((p) => ({ ...p, assistant: data.text }));
              if (data.final && startTimeRef.current) {
                addLog('info', `Assistant: ${data.text}`);
                startTimeRef.current = null;
              }
              break;
            case 'conversation_history':
              setConversationHistory(data.history);
              addLog('debug', `Received ${data.history.length} history items`);
              break;
            case 'system_prompt':
              setSystemPrompt(data.prompt);
              setTempSystemPrompt(data.prompt);
              addLog('info', 'System prompt received from backend');
              break;
            case 'stt_service':
              setSttService(data.service);
              addLog('info', `STT service set to ${data.service}`);
              break;
            case 'llm_model':
              setLlmModel(data.model);
              addLog('info', `LLM model set to ${data.model}`);
              break;
            case 'llm_service':
              setLlmService(data.service);
              addLog('info', `LLM service set to ${data.service}`);
              break;
            case 'tts_service':
              setTtsService(data.service);
              addLog('info', `TTS service set to ${data.service}`);
              break;
            case 'latency_update':
              setLatencyData(data.latencies);
              addLog(
                'debug',
                `Latencies — STT: ${data.latencies.stt}ms, LLM: ${data.latencies.llm}ms, TTS: ${data.latencies.tts}ms, Total: ${data.latencies.total}ms`,
              );
              break;
            case 'notification':
              setNotification(data.message);
              addLog('info', data.message);
              break;
            case 'audio_connected':
              // Audio connection status from backend
              if (!data.status && isRecording) {
                // Audio disconnected, stop recording
                stopRecording();
              }
              break;
            case 'log':
              if (data.message && !data.message.includes('Sent to frontend')) {
                addLog(data.level, data.message);
              }
              break;
            default:
              break;
          }
        } catch (err) {
          console.error('Error parsing data message:', err);
          addLog('error', `Error parsing message: ${err.message}`);
        }
      };
    } catch (err) {
      console.error('Failed to create data WS', err);
      addLog('error', `Failed to create WebSocket: ${err.message}`);
      setNotification('Failed to connect to backend');
      setTimeout(connectDataWebSocket, 3000);
    }
  };

  // ---------------------------------------------------------------------------
  // WebSocket: AUDIO channel + recording --------------------------------------
  // ---------------------------------------------------------------------------
  const startRecording = async () => {
    if (!frameTypeRef.current) {
      addLog('error', 'Protobuf not loaded yet');
      return;
    }

    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        latencyHint: 'interactive',
        sampleRate: SAMPLE_RATE,
      });

      // reset interaction view
      setCurrentInteraction({ user: '', assistant: '' });
      playTimeRef.current = 0;
      lastMessageTimeRef.current = 0;
      audioChunkCountRef.current = 0;

      if (!dataWsRef.current || dataWsRef.current.readyState !== WebSocket.OPEN) {
        connectDataWebSocket();
      }

      addLog('info', `Connecting to Audio WebSocket: ${AUDIO_WS_URL}`);
      const ws = new WebSocket(AUDIO_WS_URL);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = async () => {
        addLog('info', 'Audio WebSocket connected to FastAPI');
        
        // Get model info and format logging
        const modelInfo = AVAILABLE_MODELS.find(m => m.id === llmModel);
        const modelName = modelInfo ? modelInfo.name : llmModel;
        const serviceName = llmService === 'deepinfra' ? 'DeepInfra' : 'OpenAI';
        
        addLog('info', `Using ${sttService.toUpperCase()} STT, ${modelName} (${serviceName}), ${ttsService.toUpperCase()} TTS`);
        setIsConnected(true);

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              sampleRate: SAMPLE_RATE,
              channelCount: NUM_CHANNELS,
              autoGainControl: true,
              echoCancellation: true,
              noiseSuppression: true,
            },
          });

          addLog('info', 'Microphone access granted');
          mediaStreamRef.current = stream;

          scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(512, 1, 1);
          sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
          sourceRef.current.connect(scriptProcessorRef.current);
          scriptProcessorRef.current.connect(audioContextRef.current.destination);

          scriptProcessorRef.current.onaudioprocess = (e) => {
            if (ws.readyState !== WebSocket.OPEN) return;

            const pcmInt16 = convertFloat32ToS16PCM(e.inputBuffer.getChannelData(0));
            const pcmBytes = new Uint8Array(pcmInt16.buffer);

            const frame = frameTypeRef.current.create({
              audio: {
                audio: pcmBytes,
                sampleRate: SAMPLE_RATE,
                numChannels: NUM_CHANNELS,
              },
            });

            ws.send(frameTypeRef.current.encode(frame).finish());
          };

          setIsRecording(true);
          addLog('info', 'Recording started');
        } catch (err) {
          console.error('Microphone error', err);
          addLog('error', `Microphone access denied: ${err.message}`);
          ws.close();
        }
      };

      ws.onmessage = (e) => enqueueAudioFromProto(e.data);
      ws.onerror = (e) => {
        console.error('Audio WS error', e);
        addLog('error', 'Audio WebSocket error');
      };
      ws.onclose = () => {
        addLog('info', 'Audio WebSocket closed');
        stopRecording();
        setIsConnected(false);
      };
    } catch (err) {
      console.error('Failed to start recording', err);
      addLog('error', `Failed to start recording: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (isRecording) addLog('info', 'Stopping recording…');

    scriptProcessorRef.current?.disconnect();
    sourceRef.current?.disconnect();

    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());

    wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.close();

    scriptProcessorRef.current = null;
    sourceRef.current = null;
    mediaStreamRef.current = null;
    wsRef.current = null;

    setIsRecording(false);
    if (isRecording) addLog('info', 'Recording stopped');
  };

  // ---------------------------------------------------------------------------
  // Backend commands ----------------------------------------------------------
  // ---------------------------------------------------------------------------
  const sendBackend = (payload) => {
    if (dataWsRef.current?.readyState === WebSocket.OPEN) {
      dataWsRef.current.send(JSON.stringify(payload));
      return true;
    }
    addLog('error', 'Data WebSocket not connected');
    setNotification('Backend connection lost. Attempting reconnect…');
    connectDataWebSocket();
    return false;
  };

  const updateSystemPrompt = () => {
    if (sendBackend({ type: 'update_system_prompt', prompt: tempSystemPrompt })) {
      setSystemPrompt(tempSystemPrompt);
      setIsPromptLocked(true);
      addLog('info', 'System prompt updated + history cleared');
    }
  };

  const clearHistory = () => sendBackend({ type: 'clear_history' }) && addLog('info', 'Conversation history cleared');
  const handleSttServiceChange = (s) => sendBackend({ type: 'change_stt_service', service: s }) && setSttService(s);
  
  const handleLlmModelChange = (modelId) => {
    const model = AVAILABLE_MODELS.find(m => m.id === modelId);
    if (model) {
      sendBackend({ type: 'change_llm_model', model: modelId, service: model.service });
      setLlmModel(modelId);
      setLlmService(model.service);
    }
  };
  
  const handleTtsServiceChange = (s) => sendBackend({ type: 'change_tts_service', service: s }) && setTtsService(s);

  // ---------------------------------------------------------------------------
  // New Session Handler -------------------------------------------------------
  // ---------------------------------------------------------------------------
  const startNewSession = async () => {
    try {
      // Get the base URL (http/https version for API calls)
      const apiBaseUrl = WS_BASE_URL.replace('ws://', 'http://').replace('wss://', 'https://');
      
      // Call the FastAPI endpoint to create a new session
      const response = await fetch(`${apiBaseUrl}/api/session/new`);
      const data = await response.json();
      
      if (data.session_id) {
        // Store new session ID and reload
        sessionStorage.setItem('voice-assistant-session-id', data.session_id);
        window.location.reload();
      }
    } catch (err) {
      addLog('error', 'Failed to create new session');
      // Fallback to client-side session creation
      sessionStorage.removeItem('voice-assistant-session-id');
      window.location.reload();
    }
  };

  // ---------------------------------------------------------------------------
  // Cleanup on unmount --------------------------------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => () => {
    stopRecording();
    audioContextRef.current?.state !== 'closed' && audioContextRef.current?.close();
  }, []);

  // ---------------------------------------------------------------------------
  // JSX -----------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  return (
    <div className="app">
      {/* HEADER -------------------------------------------------------------- */}
      <header className="app-header">
        <h1>Voice Assistant Dev Testing</h1>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ fontSize: '0.75rem', color: '#888' }}>
            Session: {sessionId.substring(0, 8)}...
          </div>
          <button
            onClick={startNewSession}
            style={{
              background: '#666',
              border: 'none',
              color: '#fff',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            New Session
          </button>
          <button
            onClick={() => setShowDebugToolkit(true)}
            style={{
              background: '#4a9eff',
              border: 'none',
              color: '#fff',
              padding: '0.5rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            🛠️ Debug Toolkit
          </button>
          <div className={`connection-status ${isDataConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot" /> Backend {isDataConnected ? 'Connected' : 'Disconnected'}
          </div>
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot" /> Audio {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {notification && <div className="notification-bar">{notification}</div>}

      {/* MAIN LAYOUT --------------------------------------------------------- */}
      <div className="app-content">
        {/* LEFT PANEL ======================================================= */}
        <div className="left-panel">
          {/* System prompt -------------------------------------------------- */}
          <section className="system-prompt-section">
            <div className="section-header">
              <h2>SYSTEM PROMPT</h2>
              <button
                className={`lock-button ${isPromptLocked ? 'locked' : 'unlocked'}`}
                onClick={() => setIsPromptLocked((p) => !p)}
              >
                {isPromptLocked ? '🔒 Locked' : '🔓 Unlocked'}
              </button>
            </div>
            <textarea
              className="system-prompt-input"
              value={tempSystemPrompt}
              onChange={(e) => setTempSystemPrompt(e.target.value)}
              disabled={isPromptLocked}
            />
            <button
              className="update-button"
              onClick={updateSystemPrompt}
              disabled={isPromptLocked || tempSystemPrompt === systemPrompt}
            >
              Update & Clear History
            </button>
          </section>

          {/* STT dropdown ---------------------------------------------------- */}
          <section className="stt-service-section">
            <label className="service-dropdown">
              <span className="service-dropdown__label">STT Service</span>
              <select
                className="service-dropdown__select"
                value={sttService}
                onChange={(e) => handleSttServiceChange(e.target.value)}
                disabled={isRecording}
              >
                <option value="openai">OpenAI Whisper</option>
                <option value="deepgram">Deepgram Nova-2</option>
              </select>
            </label>
          </section>

          {/* LLM dropdown ---------------------------------------------------- */}
          <section className="llm-model-section">
            <label className="service-dropdown">
              <span className="service-dropdown__label">LLM Model</span>
              <select
                className="service-dropdown__select"
                value={llmModel}
                onChange={(e) => handleLlmModelChange(e.target.value)}
                disabled={isRecording}
                style={{ fontSize: '0.75rem', padding: '0.5rem' }}
              >
                <optgroup label="OpenAI">
                  {AVAILABLE_MODELS.filter(m => m.category === 'OpenAI').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Meta Llama">
                  {AVAILABLE_MODELS.filter(m => m.category === 'Meta Llama').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="DeepSeek">
                  {AVAILABLE_MODELS.filter(m => m.category === 'DeepSeek').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Qwen">
                  {AVAILABLE_MODELS.filter(m => m.category === 'Qwen').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Google">
                  {AVAILABLE_MODELS.filter(m => m.category === 'Google').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Microsoft">
                  {AVAILABLE_MODELS.filter(m => m.category === 'Microsoft').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
                <optgroup label="Mistral">
                  {AVAILABLE_MODELS.filter(m => m.category === 'Mistral').map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </optgroup>
              </select>
            </label>
          </section>

          {/* TTS dropdown ---------------------------------------------------- */}
          <section className="tts-service-section">
            <label className="service-dropdown">
              <span className="service-dropdown__label">TTS Service</span>
              <select
                className="service-dropdown__select"
                value={ttsService}
                onChange={(e) => handleTtsServiceChange(e.target.value)}
                disabled={isRecording}
              >
                <option value="elevenlabs">ElevenLabs</option>
                <option value="deepgram">Deepgram Aura</option>
              </select>
            </label>
          </section>

          {/* Record button --------------------------------------------------- */}
          <section className="recording-section">
            <button
              className={`record-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!frameTypeRef.current || !isDataConnected}
            >
              {isRecording ? (
                <>
                  <span className="stop-icon">⏹</span> Stop Recording
                </>
              ) : (
                <>
                  <span className="mic-icon">🎤</span> Start Recording
                </>
              )}
            </button>
          </section>

          {/* Latencies ------------------------------------------------------- */}
          <section className="latencies-section">
            <h2>CURRENT LATENCIES</h2>
            <div className="latency-grid">
              <div className="latency-item">
                <div className="latency-label">STT</div>
                <div className="latency-value">{latencyData.stt}ms</div>
              </div>
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

          {/* Current interaction -------------------------------------------- */}
          <section className="current-interaction-section">
            <h2>Current Interaction</h2>
            <div className="interaction-content">
              <div className="interaction-item">
                <div className="interaction-label">USER</div>
                <div className="interaction-text">{currentInteraction.user || 'Waiting for input…'}</div>
              </div>
              <div className="interaction-item">
                <div className="interaction-label">ASSISTANT</div>
                <div className="interaction-text">{currentInteraction.assistant || 'Waiting for response…'}</div>
              </div>
            </div>
          </section>
        </div>

        {/* MIDDLE PANEL ======================================================= */}
        <div className="middle-panel">
          <section className="conversation-history-section">
            <div className="section-header">
              <h2>CONVERSATION HISTORY</h2>
              <button className="clear-button" onClick={clearHistory}>Clear</button>
            </div>
            <div className="conversation-list">
              {conversationHistory.length === 0 ? (
                <div
                  className="history-message"
                  style={{ textAlign: 'center', color: '#666', padding: '2rem' }}
                >
                  No conversation history yet. Start recording to begin.
                </div>
              ) : (
                conversationHistory.map((msg, idx) => (
                  <div key={idx} className={`history-message ${msg.role}`}>
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

        {/* RIGHT PANEL ======================================================= */}
        <div className="right-panel">
          <section className="debug-logs-section">
            <div className="section-header">
              <h2>DEBUG LOGS</h2>
              <button className="clear-button" onClick={() => setLogs([])}>Clear</button>
            </div>
            <div className="logs-container">
              {logs.map((log, idx) => (
                <div key={idx} className={`log-entry ${log.level}`}>
                  <span className="log-timestamp">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </section>
        </div>
      </div>

      {/* Debug Toolkit Modal */}
      {showDebugToolkit && (
        <AudioDebugToolkit onClose={() => setShowDebugToolkit(false)} />
      )}
    </div>
  );
}