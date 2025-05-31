import { useState, useRef, useEffect } from 'react';
import protobuf from 'protobufjs';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [botReply, setBotReply] = useState('');
  const [status, setStatus] = useState('Loading protobuf...');
  
  const wsRef = useRef(null);
  const dataWsRef = useRef(null); // New: WebSocket for data/transcriptions
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
      }
    } catch (err) {
      console.error('Error processing frame:', err);
    }
  };
  
  const connectDataWebSocket = () => {
    // Connect to the data WebSocket for transcriptions and responses
    const dataWs = new WebSocket('ws://localhost:8766');
    dataWsRef.current = dataWs;
    
    dataWs.onopen = () => {
      console.log('Data WebSocket connected');
    };
    
    dataWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'connection') {
          console.log('Frontend connection confirmed:', data.status);
        } else if (data.type === 'transcription') {
          console.log('Transcription:', data.text, data.final ? '(final)' : '(interim)');
          if (data.final) {
            setTranscript(data.text);
            setStatus('Processing...');
          } else {
            setTranscript(data.text + '...');
          }
        } else if (data.type === 'assistant_reply') {
          console.log('Assistant:', data.text, data.final ? '(final)' : '(partial)');
          setBotReply(data.text);
          setStatus(data.final ? 'Assistant replied' : 'Assistant speaking...');
        }
      } catch (err) {
        console.error('Error parsing data message:', err);
      }
    };
    
    dataWs.onerror = (error) => {
      console.error('Data WebSocket error:', error);
    };
    
    dataWs.onclose = () => {
      console.log('Data WebSocket closed');
      dataWsRef.current = null;
    };
    
    return dataWs;
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
      
      // Connect data WebSocket first
      connectDataWebSocket();
      
      // Connect audio WebSocket
      const ws = new WebSocket('ws://localhost:8765');
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;
      
      setStatus('Connecting...');
      
      ws.onopen = async () => {
        console.log('Audio WebSocket connected');
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
        console.error('Audio WebSocket error:', error);
        setStatus('Connection error');
      };
      
      ws.onclose = () => {
        console.log('Audio WebSocket closed');
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
    
    // Close both WebSockets
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;
    
    if (dataWsRef.current && dataWsRef.current.readyState === WebSocket.OPEN) {
      dataWsRef.current.close();
    }
    dataWsRef.current = null;
    
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
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif', maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Voice Assistant (Pipecat)</h1>
        <div style={{
          padding: '0.5rem 1rem',
          borderRadius: '4px',
          backgroundColor: isConnected ? '#065f46' : '#7f1d1d',
          color: 'white',
          display: 'inline-block'
        }}>
          {status}
        </div>
      </header>
      
      <main>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!frameTypeRef.current}
            style={{
              backgroundColor: isRecording ? '#dc2626' : '#2563eb',
              color: 'white',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '8px',
              fontSize: '1.125rem',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            {isRecording ? (
              <>
                <span style={{
                  width: '8px',
                  height: '8px',
                  backgroundColor: 'white',
                  borderRadius: '50%',
                  animation: 'pulse 1.5s infinite'
                }}></span>
                Stop Recording
              </>
            ) : (
              <>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="3" width="6" height="11" rx="3"/>
                  <path d="M5 12v1a7 7 0 0014 0v-1M12 18v3"/>
                </svg>
                Start Recording
              </>
            )}
          </button>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '1rem',
            borderRadius: '8px',
            borderLeft: '4px solid #2563eb'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', color: '#1f2937' }}>You</div>
            <div style={{ color: '#4b5563' }}>
              {transcript || <span style={{ fontStyle: 'italic' }}>Speak into your microphone...</span>}
            </div>
          </div>
          
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '1rem',
            borderRadius: '8px',
            borderLeft: '4px solid #10b981'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', color: '#1f2937' }}>Assistant</div>
            <div style={{ color: '#4b5563' }}>
              {botReply || <span style={{ fontStyle: 'italic' }}>Waiting for response...</span>}
            </div>
          </div>
        </div>
        
        <div style={{
          marginTop: '2rem',
          padding: '1rem',
          backgroundColor: '#1f2937',
          color: '#d1d5db',
          borderRadius: '8px',
          fontSize: '0.875rem'
        }}>
          <div style={{ marginBottom: '0.5rem', fontWeight: 'bold' }}>Debug Info:</div>
          <div>Recording: {isRecording ? 'Yes' : 'No'}</div>
          <div>Connected: {isConnected ? 'Yes' : 'No'}</div>
          <div>Audio WebSocket: ws://localhost:8765</div>
          <div>Data WebSocket: ws://localhost:8766</div>
        </div>
      </main>
      
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}