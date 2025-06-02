import { useState, useRef, useEffect } from 'react';
import protobuf from 'protobufjs';
import './App.css';
import AudioDebugToolkit from './AudioDebugToolkit';
import { useAudioWorklet } from './audioWorklet/useAudioWorklet';
import { useVAD } from './hooks/useVAD';

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

// TTS Service Sample Rates
const TTS_SAMPLE_RATES = {
  elevenlabs: 24000,
  deepgram: 16000,
  openai: 24000
};


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
  const [ttsSampleRate, setTtsSampleRate] = useState(TTS_SAMPLE_RATES.elevenlabs);
  const [notification, setNotification] = useState('');
  const [showDebugToolkit, setShowDebugToolkit] = useState(false);

  // VAD States
  const [vadEnabled, setVadEnabled] = useState(true);  // Re-enable VAD
  const [vadMode, setVadMode] = useState(2);
  const [vadActivity, setVadActivity] = useState({ 
    isSpeaking: false, 
    energy: 0, 
    threshold: 0.01 
  });

  // Performance Metrics
  const [chunkingMetrics, setChunkingMetrics] = useState({
    timeToFirstAudio: 0,
    chunksReceived: 0
  });
  const [vadStats, setVadStats] = useState({
    trafficReduction: 0,
    accuracyImprovement: 0
  });

  // ---------------------------------------------------------------------------
  // Refs ----------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const wsRef = useRef(null); // audio WS
  const dataWsRef = useRef(null); // data WS
  const frameTypeRef = useRef(null);
  const logsEndRef = useRef(null);
  const conversationEndRef = useRef(null);
  const startTimeRef = useRef(null);
  const audioChunkCountRef = useRef(0);
  const reconnectTimeoutRef = useRef(null);
  const audioWorkletRef = useRef(null);
  const totalAudioFramesRef = useRef(0);
  const sentAudioFramesRef = useRef(0);

  // ---------------------------------------------------------------------------
  // Constants -----------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const SAMPLE_RATE = 16000;
  const NUM_CHANNELS = 1;
  
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
  // console.log('WebSocket Configuration:', {
  //   BASE_URL: WS_BASE_URL,
  //   DATA_URL: DATA_WS_URL,
  //   AUDIO_URL: AUDIO_WS_URL,
  //   ENV_URL: import.meta.env.VITE_WS_URL
  // });

  // ---------------------------------------------------------------------------
  // AudioWorklet Hook ---------------------------------------------------------
  // ---------------------------------------------------------------------------
  const currentAudioFrameRef = useRef(null);
  const audioFrameCountRef = useRef(0);
  
  const {
    startRecording: startAudioWorklet,
    stopRecording: stopAudioWorklet,
    queueAudioForPlayback,
    isRecording: isWorkletRecording
  } = useAudioWorklet({
    sampleRate: SAMPLE_RATE,
    outputSampleRate: ttsSampleRate,
    onAudioData: (audioData) => {
      audioFrameCountRef.current++;
      
      // Log every 100th frame to verify audio is coming through
      if (audioFrameCountRef.current % 100 === 0) {
        addLog('debug', `AudioWorklet: Received ${audioFrameCountRef.current} frames, current frame size: ${audioData.length} bytes`);
      }
      
      if (!vadEnabled) {
        // VAD disabled, send all audio
        sendAudioFrame(audioData);
      } else {
        // Store the current PCM frame
        currentAudioFrameRef.current = audioData;
        totalAudioFramesRef.current++;
        
        // Convert PCM back to Float32 for VAD processing
        const float32Data = convertS16PCMToFloat32(audioData);
        
        // Debug: Check if we're getting valid audio data
        if (audioFrameCountRef.current === 1) {
          const maxValue = Math.max(...float32Data.map(Math.abs));
          addLog('debug', `VAD: First audio frame max amplitude: ${maxValue.toFixed(4)}`);
        }
        
        vadProcessor.processAudioData(float32Data);
      }
    }
  });

  // ---------------------------------------------------------------------------
  // VAD Hook ------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const vadProcessor = useVAD({
    sampleRate: SAMPLE_RATE,
    mode: vadMode,
    onSpeechStart: (data) => {
      addLog('info', 'VAD: Speech detected, starting transmission');
      // Send buffered audio when speech starts
      if (wsRef.current?.readyState === WebSocket.OPEN && data.preSpeechData) {
        // Split the pre-speech buffer into chunks and send
        const chunkSize = 512; // Match the AudioWorklet buffer size
        for (let i = 0; i < data.preSpeechData.length; i += chunkSize) {
          const chunk = data.preSpeechData.slice(i, Math.min(i + chunkSize, data.preSpeechData.length));
          const pcmBytes = convertFloat32ToS16PCM(chunk);
          sendAudioFrame(new Uint8Array(pcmBytes.buffer));
        }
      }
    },
    onSpeechEnd: (data) => {
      addLog('info', `VAD: Speech ended after ${data.duration}ms`);
      // Update traffic reduction stats
      const reduction = totalAudioFramesRef.current > 0 
        ? Math.round((1 - sentAudioFramesRef.current / totalAudioFramesRef.current) * 100)
        : 0;
      setVadStats(prev => ({ ...prev, trafficReduction: reduction }));
    },
    onVoiceActivity: (data) => {
      setVadActivity(data);
    },
    onProcessedAudio: (float32Frame) => {
      // This is called for each frame during speech
      // Send the current PCM frame we stored
      if (wsRef.current?.readyState === WebSocket.OPEN && currentAudioFrameRef.current) {
        sendAudioFrame(currentAudioFrameRef.current);
        sentAudioFramesRef.current++;
      }
    }
  });

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

  const convertS16PCMToFloat32 = (pcmData) => {
    // Create Int16Array from the Uint8Array buffer
    const int16Array = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength / 2);
    const float32Array = new Float32Array(int16Array.length);
    
    for (let i = 0; i < int16Array.length; i++) {
      // Convert from int16 range (-32768 to 32767) to float32 range (-1 to 1)
      float32Array[i] = int16Array[i] / 32768.0;
    }
    
    return float32Array;
  };

  // ---------------------------------------------------------------------------
  // Helper: Send audio frame --------------------------------------------------
  // ---------------------------------------------------------------------------
  const sendAudioFrame = (audioData) => {
    if (frameTypeRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
      const frame = frameTypeRef.current.create({
        audio: {
          audio: audioData,
          sampleRate: SAMPLE_RATE,
          numChannels: NUM_CHANNELS,
        },
      });
      wsRef.current.send(frameTypeRef.current.encode(frame).finish());
    }
  };

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
  // Effect: Update VAD mode when changed --------------------------------------
  // ---------------------------------------------------------------------------
  useEffect(() => {
    vadProcessor.setMode(vadMode);
  }, [vadMode]);

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
  // Helper: enqueue audio frames (using AudioWorklet) -------------------------
  // ---------------------------------------------------------------------------
  const enqueueAudioFromProto = (arrayBuffer) => {
    if (!frameTypeRef.current) return;

    try {
      const parsedFrame = frameTypeRef.current.decode(new Uint8Array(arrayBuffer));

      if (parsedFrame?.audio) {
        const audioArray = new Uint8Array(parsedFrame.audio.audio);
        queueAudioForPlayback(audioArray);
        addLog('debug', 'Audio chunk queued for playback');
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
        
        // Send initial VAD settings
        ws.send(JSON.stringify({
          type: 'change_vad_settings',
          enabled: vadEnabled,
          mode: vadMode
        }));
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
            case 'tts_sample_rate':
              setTtsSampleRate(data.sample_rate);
              addLog('info', `TTS sample rate set to ${data.sample_rate}Hz`);
              break;
            case 'latency_update':
              setLatencyData(data.latencies);
              addLog(
                'debug',
                `Latencies — STT: ${data.latencies.stt}ms, LLM: ${data.latencies.llm}ms, TTS: ${data.latencies.tts}ms, Total: ${data.latencies.total}ms`,
              );
              break;
            case 'metric':
              if (data.metric === 'time_to_first_audio') {
                setChunkingMetrics(prev => ({
                  ...prev,
                  timeToFirstAudio: data.value
                }));
                addLog('info', `Time to first audio: ${data.value}ms`);
              }
              break;
            case 'vad_settings':
              setVadEnabled(data.enabled);
              setVadMode(data.mode);
              addLog('info', `VAD settings updated: ${data.enabled ? 'enabled' : 'disabled'}, mode ${data.mode}`);
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
      // reset interaction view
      setCurrentInteraction({ user: '', assistant: '' });
      audioChunkCountRef.current = 0;
      totalAudioFramesRef.current = 0;
      sentAudioFramesRef.current = 0;

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
        
        // Format TTS service info
        let ttsInfo = ttsService.toUpperCase();
        if (ttsService === 'openai') {
          ttsInfo = 'OpenAI (Nova)';
        }
        
        addLog('info', `Using ${sttService.toUpperCase()} STT, ${modelName} (${serviceName}), ${ttsInfo} TTS (${ttsSampleRate}Hz)`);
        addLog('info', `VAD: ${vadEnabled ? `Enabled (mode ${vadMode})` : 'Disabled'}`);
        setIsConnected(true);

        try {
          // Start VAD if enabled
          if (vadEnabled) {
            vadProcessor.startVAD();
          }
          
          // Use AudioWorklet instead of ScriptProcessor
          const result = await startAudioWorklet();
          if (result) {
            audioWorkletRef.current = result;
            setIsRecording(true);
            addLog('info', 'Recording started with AudioWorklet');
          }
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

    // Stop VAD
    vadProcessor.stopVAD();
    
    // Stop AudioWorklet
    stopAudioWorklet();

    wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.close();
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
  
  const handleTtsServiceChange = (s) => {
    if (sendBackend({ type: 'change_tts_service', service: s })) {
      setTtsService(s);
      // Update sample rate immediately on frontend
      setTtsSampleRate(TTS_SAMPLE_RATES[s] || 24000);
    }
  };

  const handleVadEnabledChange = (enabled) => {
    setVadEnabled(enabled);
    sendBackend({ type: 'change_vad_settings', enabled, mode: vadMode });
    if (isRecording) {
      enabled ? vadProcessor.startVAD() : vadProcessor.stopVAD();
    }
  };

  const handleVadModeChange = (mode) => {
    setVadMode(mode);
    vadProcessor.setMode(mode);
    sendBackend({ type: 'change_vad_settings', enabled: vadEnabled, mode });
  };

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
              <span className="service-dropdown__label">TTS Service ({ttsSampleRate}Hz)</span>
              <select
                className="service-dropdown__select"
                value={ttsService}
                onChange={(e) => handleTtsServiceChange(e.target.value)}
                disabled={isRecording}
              >
                <option value="elevenlabs">ElevenLabs (24000Hz)</option>
                <option value="deepgram">Deepgram Aura (16000Hz)</option>
                <option value="openai">OpenAI Nova (24000Hz)</option>
              </select>
            </label>
          </section>

          {/* VAD Settings ---------------------------------------------------- */}
          <section className="vad-settings-section">
            <h2>Voice Activity Detection</h2>
            <label className="switch" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <input
                type="checkbox"
                checked={vadEnabled}
                onChange={(e) => handleVadEnabledChange(e.target.checked)}
                style={{ width: '20px', height: '20px' }}
              />
              <span>VAD {vadEnabled ? 'Enabled' : 'Disabled'}</span>
            </label>
            
            <label className="service-dropdown">
              <span className="service-dropdown__label">VAD Sensitivity</span>
              <select
                className="service-dropdown__select"
                value={vadMode}
                onChange={(e) => handleVadModeChange(parseInt(e.target.value))}
                disabled={!vadEnabled}
              >
                <option value="0">Very Permissive</option>
                <option value="1">Permissive</option>
                <option value="2">Balanced</option>
                <option value="3">Aggressive</option>
              </select>
            </label>
            
            {/* Visual VAD indicator */}
            <div className="vad-indicator" style={{ marginTop: '1rem' }}>
              <div className={`vad-status ${vadActivity.isSpeaking ? 'speaking' : 'silent'}`} 
                   style={{ 
                     padding: '0.5rem', 
                     textAlign: 'center', 
                     borderRadius: '4px',
                     background: vadActivity.isSpeaking ? '#4CAF50' : '#666',
                     color: '#fff',
                     marginBottom: '0.5rem'
                   }}>
                {vadActivity.isSpeaking ? '🎤 Speaking' : '🔇 Silent'}
              </div>
              <div className="vad-energy-bar" style={{ position: 'relative', height: '20px', background: '#333', borderRadius: '4px' }}>
                <div 
                  className="vad-energy-level" 
                  style={{ 
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${Math.min(vadActivity.energy * 500, 100)}%`,
                    background: '#4a9eff',
                    borderRadius: '4px',
                    transition: 'width 0.1s'
                  }}
                />
                <div 
                  className="vad-threshold" 
                  style={{ 
                    position: 'absolute',
                    left: `${Math.min(vadActivity.threshold * 500, 100)}%`,
                    top: 0,
                    width: '2px',
                    height: '100%',
                    background: '#ff4a4a'
                  }}
                />
              </div>
              {vadEnabled && (
                <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.5rem' }}>
                  Traffic Reduction: {vadStats.trafficReduction}%
                </div>
              )}
            </div>
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

          {/* Performance Metrics --------------------------------------------- */}
          <section className="performance-metrics">
            <h2>Performance Improvements</h2>
            <div className="metric-grid" style={{ display: 'grid', gap: '0.5rem', fontSize: '0.875rem' }}>
              <div className="metric-card" style={{ padding: '0.5rem', background: '#2a2a2a', borderRadius: '4px' }}>
                <h4 style={{ margin: '0 0 0.25rem 0' }}>AudioWorklet</h4>
                <p style={{ margin: 0, color: '#4CAF50' }}>✓ Active</p>
                <p style={{ margin: 0, fontSize: '0.75rem', color: '#888' }}>~50ms latency reduction</p>
              </div>
              <div className="metric-card" style={{ padding: '0.5rem', background: '#2a2a2a', borderRadius: '4px' }}>
                <h4 style={{ margin: '0 0 0.25rem 0' }}>Sentence Chunking</h4>
                {chunkingMetrics.timeToFirstAudio > 0 ? (
                  <>
                    <p style={{ margin: 0, color: '#4CAF50' }}>✓ Active</p>
                    <p style={{ margin: 0, fontSize: '0.75rem', color: '#888' }}>First audio: {chunkingMetrics.timeToFirstAudio}ms</p>
                  </>
                ) : (
                  <p style={{ margin: 0, color: '#666' }}>Waiting for data...</p>
                )}
              </div>
              <div className="metric-card" style={{ padding: '0.5rem', background: '#2a2a2a', borderRadius: '4px' }}>
                <h4 style={{ margin: '0 0 0.25rem 0' }}>Client VAD</h4>
                <p style={{ margin: 0, color: vadEnabled ? '#4CAF50' : '#666' }}>
                  {vadEnabled ? '✓ Active' : '✗ Disabled'}
                </p>
                {vadEnabled && (
                  <p style={{ margin: 0, fontSize: '0.75rem', color: '#888' }}>
                    Traffic: -{vadStats.trafficReduction}%
                  </p>
                )}
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