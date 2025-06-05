import React, { useState, useRef, useEffect, useCallback } from 'react';
import protobuf from 'protobufjs';
import './App.css';

const SAMPLE_RATE = 16000;
const CHANNELS = 1;
const CHUNK_SIZE = 320;
const SILENCE_THRESHOLD = 500;
const WS_BASE_URL = 'ws://localhost:8000';

// Pipecat protobuf schema (matching frames_pb2.py)
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
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [status, setStatus] = useState('Ready to start');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const websocketRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const protoRootRef = useRef(null);
  const frameTypeRef = useRef(null);

  // Initialize protobuf
  useEffect(() => {
    const initProtobuf = async () => {
      try {
        const root = await protobuf.parse(PIPECAT_PROTO).root;
        protoRootRef.current = root;
        frameTypeRef.current = root.lookupType('pipecat.Frame');
      } catch (error) {
        console.error('Failed to initialize protobuf:', error);
      }
    };
    initProtobuf();
  }, []);

  // Get audio devices on mount
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        setDevices(audioInputs);
        if (audioInputs.length > 0 && !selectedDevice) {
          setSelectedDevice(audioInputs[0].deviceId);
        }
      });
  }, []);

  // Get session ID
  const getSessionId = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/session/new');
      const data = await response.json();
      return data.session_id;
    } catch (error) {
      console.error('Failed to get session ID:', error);
      throw error;
    }
  };

  // Encode audio frame to protobuf
  const encodeAudioFrame = (audioData /* Uint8Array */) => {
    if (!frameTypeRef.current) {
      console.error("Protobuf not initialized");
      return null;
    }
  
    /* The JS object must use the SAME tag-order field names that
       protobuf-js generated from frames.proto:
         3 → audio        (bytes)
         4 → sampleRate   (uint32)
         5 → numChannels  (uint32)
       id (tag 1) and name (tag 2) are omitted – that’s fine in proto3. */
  
    const payload = {
      audio: {                      // <-- one-of selector (tag 2 in Frame)
        audio: audioData,           // tag 3
        sampleRate: SAMPLE_RATE,    // tag 4  (camelCase!)
        numChannels: CHANNELS,      // tag 5
      },
    };
  
    // Catch schema mismatches early
    const err = frameTypeRef.current.verify(payload);
    if (err) {
      console.error("Audio frame verification failed:", err);
      return null;
    }
  
    try {
      const message = frameTypeRef.current.create(payload);
      return frameTypeRef.current.encode(message).finish(); // Uint8Array
    } catch (error) {
      console.error("Failed to encode frame:", error);
      return null;
    }
  };
  
  
  
  const decodeFrame = (data /* ArrayBuffer */) => {
    if (!frameTypeRef.current) {
      console.error("Protobuf not initialized");
      return null;
    }
  
    try {
      // protobufjs happily accepts a Uint8Array view
      const decoded = frameTypeRef.current.decode(new Uint8Array(data));
      console.log('decoded')
      console.log(decoded)
      // Keep bytes as Uint8Array, convert uint64 to Number, include defaults
      const frame = frameTypeRef.current.toObject(decoded, {
        bytes: Uint8Array,
        longs: Number,
        defaults: true,
      });
      console.log(frame)
      return frame;
    } catch (error) {
      console.error("Failed to decode frame:", error);
      return null;
    }
  };

  // Play audio queue
  const playAudioQueue = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;
  
    isPlayingRef.current = true;
    const audioContext = audioContextRef.current;
  
    while (audioQueueRef.current.length > 0) {
      let audioData = audioQueueRef.current.shift();   // Uint8Array
  
      try {
        // -----------------------------------------------------------------
        // Ensure the view starts at an even address for Int16Array
        // -----------------------------------------------------------------
        if (audioData.byteOffset & 1) {                // odd? → realign
          audioData = new Uint8Array(audioData);       // copy, offset = 0
        }
  
        const int16Array = new Int16Array(
          audioData.buffer,
          audioData.byteOffset,                        // now guaranteed even
          audioData.byteLength >> 1
        );
  
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
          float32Array[i] = int16Array[i] / 32768;
        }
  
        const audioBuffer = audioContext.createBuffer(
          1,
          float32Array.length,
          SAMPLE_RATE
        );
        audioBuffer.getChannelData(0).set(float32Array);
  
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
  
        await new Promise((resolve) => {
          source.onended = resolve;
          source.start();
        });
      } catch (error) {
        console.error("Audio playback error:", error);
      }
    }
  
    isPlayingRef.current = false;
    setIsAssistantSpeaking(false);
    setStatus("Ready for next input");
  };
  

  // Process audio worklet
  const setupAudioWorklet = async (stream) => {
    const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioContextRef.current = audioContext;

    // Create worklet processor code
    const processorCode = `
      class AudioProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.bufferSize = 320;
          this.buffer = [];
        }

        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input[0]) {
            const samples = input[0];
            
            for (let i = 0; i < samples.length; i++) {
              this.buffer.push(samples[i]);
              
              if (this.buffer.length >= this.bufferSize) {
                const int16Buffer = new Int16Array(this.bufferSize);
                for (let j = 0; j < this.bufferSize; j++) {
                  const s = Math.max(-1, Math.min(1, this.buffer[j]));
                  int16Buffer[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                this.port.postMessage({
                  type: 'audio',
                  data: int16Buffer.buffer
                });
                
                this.buffer = [];
              }
            }
          }
          return true;
        }
      }
      registerProcessor('audio-processor', AudioProcessor);
    `;

    const blob = new Blob([processorCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(url);

    const source = audioContext.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioContext, 'audio-processor');
    
    processor.port.onmessage = (event) => {
      if (event.data.type === 'audio' && !isAssistantSpeaking && websocketRef.current?.readyState === WebSocket.OPEN) {
        // Encode and send audio frame
        const encoded = encodeAudioFrame(new Uint8Array(event.data.data));
        if (encoded) {
          websocketRef.current.send(encoded);
        }
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    processorRef.current = processor;
  };

  // Start conversation
  const startConversation = async () => {
    try {
      setStatus('Getting session ID...');
      const newSessionId = await getSessionId();
      setSessionId(newSessionId);

      setStatus('Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedDevice,
          channelCount: CHANNELS,
          sampleRate: SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      mediaStreamRef.current = stream;

      setStatus('Setting up audio processing...');
      await setupAudioWorklet(stream);

      setStatus('Connecting to server...');
      const ws = new WebSocket(`${WS_BASE_URL}/ws/audio?session=${newSessionId}`);
      websocketRef.current = ws;

      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setIsConnected(true);
        setIsRecording(true);
        setStatus('Connected! Speak naturally...');
      };

      ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
          try {
            const frame = decodeFrame(event.data);
            
            if (frame) {
              if (frame.audio) {
                if (!isAssistantSpeaking) {
                  setIsAssistantSpeaking(true);
                  setStatus('Assistant speaking...');
                }
                
                // The audio field is already a Uint8Array from protobufjs
                audioQueueRef.current.push(frame.audio.audio);
                playAudioQueue();
              } else if (frame.text) {
                // Handle text response from assistant
                console.log('Assistant:', frame.text.text);
                setConversationHistory(prev => [...prev, { 
                  speaker: 'assistant', 
                  text: frame.text.text 
                }]);
              } else if (frame.transcription) {
                // Handle transcription of user speech
                console.log('User said:', frame.transcription.text);
                setConversationHistory(prev => [...prev, { 
                  speaker: 'user', 
                  text: frame.transcription.text 
                }]);
                
                // Check for exit commands
                const text = frame.transcription.text.toLowerCase();
                if (['goodbye', 'bye', 'exit', 'quit'].some(word => text.includes(word))) {
                  setStatus('Goodbye! Thanks for chatting!');
                  setTimeout(() => stopConversation(), 2000);
                }
              }
            }
          } catch (error) {
            console.error('Message parsing error:', error);
          }
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error');
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsRecording(false);
        setStatus('Disconnected');
      };

    } catch (error) {
      console.error('Failed to start:', error);
      setStatus(`Error: ${error.message}`);
    }
  };

  // Stop conversation
  const stopConversation = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    setIsRecording(false);
    setIsConnected(false);
    setStatus('Conversation ended');
  };

  return (
    <div className="app">
      <div className="container">
        <h1>🎙️ Interactive Voice Assistant</h1>
        
        <div className="status-bar">
          <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
            {status}
          </div>
          {sessionId && (
            <div className="session-id">Session: {sessionId}</div>
          )}
        </div>

        <div className="controls">
          <div className="device-selector">
            <label>Audio Device:</label>
            <select 
              value={selectedDevice} 
              onChange={(e) => setSelectedDevice(e.target.value)}
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
          <div className={`indicator ${isRecording ? 'active' : ''}`}>
            🎤 {isRecording ? 'Recording' : 'Not Recording'}
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