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
    isMicMuted: false
  });

  const refs = useRef({
    audioContext: null,
    mediaStream: null,
    processor: null,
    websocket: null,
    audioQueue: [],
    isPlaying: false,
    protoRoot: null,
    frameType: null,
    isAssistantSpeaking: false
  });

  // State helper
  const updateState = (updates) => setState(prev => ({ ...prev, ...updates }));

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

  // Get audio devices
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(devices => {
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      updateState({ 
        devices: audioInputs,
        selectedDevice: audioInputs[0]?.deviceId || ''
      });
    });
  }, []);

  // Sync assistant speaking state
  useEffect(() => {
    refs.current.isAssistantSpeaking = state.isAssistantSpeaking;
  }, [state.isAssistantSpeaking]);

  // Audio encoding/decoding
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

  // Audio playback
  const playAudioQueue = async () => {
    const { audioQueue, audioContext, isPlaying } = refs.current;
    if (isPlaying || !audioQueue.length) return;

    refs.current.isPlaying = true;
    updateState({ isAssistantSpeaking: true, isMicMuted: true });

    while (audioQueue.length > 0) {
      const { bytes: audioData, rate } = audioQueue.shift();

      try {
        const aligned = audioData.byteOffset & 1 ? new Uint8Array(audioData) : audioData;
        const int16 = new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength >> 1);
        const float32 = new Float32Array(int16.length);
        
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768;
        }

        const buffer = audioContext.createBuffer(1, float32.length, rate);
        buffer.getChannelData(0).set(float32);

        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);

        await new Promise(resolve => {
          source.onended = resolve;
          source.start();
        });
      } catch (error) {
        console.error("Playback error:", error);
      }
    }

    refs.current.isPlaying = false;
    updateState({ isAssistantSpeaking: false, isMicMuted: false, status: "Ready for next input" });
  };

  // Audio worklet setup
  const setupAudioWorklet = async (stream) => {
    const audioContext = new AudioContext({ sampleRate: CONFIG.MIC_SAMPLE_RATE });
    refs.current.audioContext = audioContext;

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
        if (encoded) refs.current.websocket.send(encoded);
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    refs.current.processor = processor;
  };

  // WebSocket message handler
  const handleWebSocketMessage = async (event) => {
    if (!(event.data instanceof ArrayBuffer)) return;

    const frame = decodeFrame(event.data);
    if (!frame) return;

    if (frame.audio) {
      updateState({ 
        isAssistantSpeaking: true, 
        isMicMuted: true, 
        status: 'Assistant speaking...' 
      });
      refs.current.audioQueue.push({ 
        bytes: frame.audio.audio, 
        rate: frame.audio.sampleRate 
      });
      playAudioQueue();
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

  // Start conversation
  const startConversation = async () => {
    try {
      updateState({ status: 'Getting session ID...' });
      const response = await fetch(`${CONFIG.API_BASE_URL}/api/session/new`);
      const { session_id } = await response.json();
      updateState({ sessionId: session_id });

      updateState({ status: 'Requesting microphone access...' });
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
      const ws = new WebSocket(`${CONFIG.WS_BASE_URL}/ws/audio?session=${session_id}`);
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => updateState({ 
        isConnected: true, 
        isRecording: true, 
        status: 'Connected! Speak naturally...' 
      });
      
      ws.onmessage = handleWebSocketMessage;
      ws.onerror = () => updateState({ status: 'Connection error' });
      ws.onclose = () => updateState({ 
        isConnected: false, 
        isRecording: false, 
        status: 'Disconnected' 
      });
      
      refs.current.websocket = ws;
    } catch (error) {
      console.error('Failed to start:', error);
      updateState({ status: `Error: ${error.message}` });
    }
  };

  // Stop conversation
  const stopConversation = () => {
    const { websocket, mediaStream, audioContext } = refs.current;
    
    websocket?.close();
    mediaStream?.getTracks().forEach(track => track.stop());
    audioContext?.close();

    updateState({
      isRecording: false,
      isConnected: false,
      isAssistantSpeaking: false,
      isMicMuted: false,
      status: 'Conversation ended'
    });
  };

  const { 
    isRecording, isConnected, isAssistantSpeaking, sessionId, 
    status, conversationHistory, devices, selectedDevice, isMicMuted 
  } = state;

  return (
    <div className="app">
      <div className="container">
        <h1>🎙️ Interactive Voice Assistant</h1>
        
        <div className="status-bar">
          <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
            {status}
          </div>
          {sessionId && <div className="session-id">Session: {sessionId}</div>}
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
        </div>

        <div className="indicators">
          <div className={`indicator ${isRecording && !isMicMuted ? 'active' : ''}`}>
            🎤 {isRecording ? (isMicMuted ? 'Mic Muted' : 'Recording') : 'Not Recording'}
          </div>
          <div className={`indicator ${isAssistantSpeaking ? 'active' : ''}`}>
            🤖 {isAssistantSpeaking ? 'Assistant Speaking' : 'Assistant Idle'}
          </div>
        </div>

        <div className="tips">
          <h3>💡 Tips:</h3>
          <ul>
            <li>Speak naturally, the assistant will respond</li>
            <li>Say 'goodbye' to end the conversation</li>
            <li>The microphone automatically mutes when the assistant speaks</li>
            <li>The conversation will continue automatically</li>
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
    </div>
  );
}

export default App;