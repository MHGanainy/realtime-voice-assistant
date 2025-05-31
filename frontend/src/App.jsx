import { useState, useRef, useEffect } from 'react';
import protobuf from 'protobufjs';
import './App.css';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [botReply, setBotReply] = useState('');
  const [status, setStatus] = useState('Loading protobuf...');
  
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const sourceRef = useRef(null);
  const frameTypeRef = useRef(null);
  const playTimeRef = useRef(0);
  const lastMessageTimeRef = useRef(0);
  
  // Constants
  const SAMPLE_RATE = 16000;
  const NUM_CHANNELS = 1;
  const PLAY_TIME_RESET_THRESHOLD_MS = 1.0;
  
  // Load protobuf schema
  useEffect(() => {
    // Define the proto schema inline
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
      setStatus('Ready! Click "Start Recording" to begin.');
    } catch (err) {
      console.error('Failed to load protobuf:', err);
      setStatus('Error loading protobuf: ' + err.message);
    }
  }, []);
  
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
        // Reset play time if it's been a while
        const currentTime = audioContextRef.current.currentTime;
        const diffTime = currentTime - lastMessageTimeRef.current;
        if (playTimeRef.current === 0 || diffTime > PLAY_TIME_RESET_THRESHOLD_MS) {
          playTimeRef.current = currentTime;
        }
        lastMessageTimeRef.current = currentTime;
        
        // Convert audio data
        const audioVector = Array.from(parsedFrame.audio.audio);
        const audioArray = new Uint8Array(audioVector);
        
        // Decode and play
        audioContextRef.current.decodeAudioData(audioArray.buffer, (buffer) => {
          const source = audioContextRef.current.createBufferSource();
          source.buffer = buffer;
          source.start(playTimeRef.current);
          source.connect(audioContextRef.current.destination);
          playTimeRef.current += buffer.duration;
        }, (error) => {
          console.error('Error decoding audio:', error);
        });
      } else if (parsedFrame?.transcription) {
        setTranscript(parsedFrame.transcription.text);
        setStatus('Processing...');
      } else if (parsedFrame?.text) {
        setBotReply(parsedFrame.text.text);
        setStatus('Assistant speaking...');
      }
    } catch (err) {
      console.error('Error processing frame:', err);
    }
  };
  
  const startRecording = async () => {
    if (!frameTypeRef.current) {
      setStatus('Protobuf not loaded yet');
      return;
    }
    
    try {
      // Initialize audio context
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        latencyHint: 'interactive',
        sampleRate: SAMPLE_RATE
      });
      
      // Clear previous state
      setTranscript('');
      setBotReply('');
      playTimeRef.current = 0;
      lastMessageTimeRef.current = 0;
      
      // Connect WebSocket
      const ws = new WebSocket('ws://localhost:8765');
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;
      
      setStatus('Connecting...');
      
      ws.onopen = async () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setStatus('Getting microphone...');
        
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
            
            // Create the frame
            const frame = {
              audio: {
                audio: pcmByteArray,
                sampleRate: SAMPLE_RATE,
                numChannels: NUM_CHANNELS
              }
            };
            
            // Encode and send
            const message = frameTypeRef.current.create(frame);
            const buffer = frameTypeRef.current.encode(message).finish();
            ws.send(buffer);
          };
          
          setIsRecording(true);
          setStatus('Listening... Speak clearly into your microphone');
          
        } catch (err) {
          console.error('Microphone error:', err);
          setStatus('Microphone access denied');
          ws.close();
        }
      };
      
      ws.onmessage = (event) => {
        enqueueAudioFromProto(event.data);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error');
      };
      
      ws.onclose = () => {
        console.log('WebSocket closed');
        stopRecording();
        setIsConnected(false);
        setStatus('Disconnected');
      };
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      setStatus('Failed to start: ' + error.message);
    }
  };
  
  const stopRecording = () => {
    console.log('Stopping recording...');
    
    // Stop audio processing
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    
    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;
    
    setIsRecording(false);
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
        <h1>Voice Assistant (Pipecat)</h1>
        <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
          {status}
        </div>
      </header>
      
      <main className="app-main">
        <div className="control-section">
          <button
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!frameTypeRef.current}
          >
            {isRecording ? (
              <>
                <span className="recording-dot"></span>
                Stop Recording
              </>
            ) : (
              <>
                <svg className="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="3" width="6" height="11" rx="3"/>
                  <path d="M5 12v1a7 7 0 0014 0v-1M12 18v3"/>
                </svg>
                Start Recording
              </>
            )}
          </button>
        </div>
        
        <div className="conversation-section">
          <div className="message user-message">
            <div className="message-label">You</div>
            <div className="message-content">
              {transcript || <span className="placeholder">Speak into your microphone...</span>}
            </div>
          </div>
          
          <div className="message assistant-message">
            <div className="message-label">Assistant</div>
            <div className="message-content">
              {botReply || <span className="placeholder">Waiting for response...</span>}
            </div>
          </div>
        </div>
        
        <div style={{marginTop: '2rem', padding: '1rem', background: '#1a1a1a', borderRadius: '8px', fontSize: '0.875rem'}}>
          <div style={{marginBottom: '0.5rem'}}>
            <strong>Debug Info:</strong>
          </div>
          <div>Recording: {isRecording ? 'Yes' : 'No'}</div>
          <div>Connected: {isConnected ? 'Yes' : 'No'}</div>
          <div>WebSocket URL: ws://localhost:8765</div>
        </div>
      </main>
    </div>
  );
}