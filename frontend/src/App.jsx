import React, { useEffect, useRef, useState } from "react";
import { VoiceRoom } from "./voice/VoiceRoom.js";
import { AudioCapture } from "./audio/AudioCapture.js";
import { AudioPlayer } from "./audio/AudioPlayer.js";
import "./App.css";

/**
 * Voice Assistant using state-of-the-art audio pipeline
 * Now with Opus compression and echo playback
 */
export default function App() {
  const [isInRoom, setIsInRoom] = useState(false);
  const [roomId, setRoomId] = useState("test-room");
  const [userId, setUserId] = useState(`user-${Math.random().toString(36).substr(2, 9)}`);
  const [transcript, setTranscript] = useState("");
  const [assistantMessage, setAssistantMessage] = useState("");
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [micLevel, setMicLevel] = useState(0);
  const [metrics, setMetrics] = useState({});
  const [lastAudioInfo, setLastAudioInfo] = useState(null);
  const [isPushToTalk, setIsPushToTalk] = useState(false);
  const [isTalking, setIsTalking] = useState(false);

  const roomRef = useRef(null);
  const captureRef = useRef(null);
  const playerRef = useRef(null);

  useEffect(() => {
    // Initialize audio components with VAD enabled
    captureRef.current = new AudioCapture({
      useOpusEncoding: true,
      enableVAD: true,
      vadGating: true, // Only send audio when voice is detected
      energyThreshold: 0.01 // Adjust this based on your microphone
    });
    
    playerRef.current = new AudioPlayer();

    return () => {
      // Cleanup on unmount
      leaveRoom();
    };
  }, []);

  const joinRoom = async () => {
    try {
      // Create voice room
      const room = new VoiceRoom({
        roomId,
        userId,
        capture: captureRef.current,
        player: playerRef.current
      });

      // Enable quality metrics for testing
      room.enableQualityMetrics = true; // Disabled by default to avoid errors

      // Setup event listeners
      room.on('connected', () => {
        setConnectionStatus('connected');
        console.log('Connected to room');
      });

      room.on('disconnected', () => {
        setConnectionStatus('disconnected');
      });

      room.on('joined', (data) => {
        console.log('Joined room:', data);
      });

      room.on('participantJoined', (data) => {
        console.log('Participant joined:', data.userId);
      });

      room.on('voiceStart', () => {
        console.log('Started speaking');
        setTranscript("🎤 Speaking...");
      });

      room.on('voiceEnd', () => {
        console.log('Stopped speaking');
        setTranscript("🔇 Silence detected");
      });

      room.on('micLevel', (level) => {
        setMicLevel(level);
      });

      room.on('audioReceived', (info) => {
        setLastAudioInfo(info);
        if (info.isOpus) {
          setAssistantMessage("Playing your voice echo (Opus compressed)");
        }
      });

      room.on('metrics', (metrics) => {
        setMetrics(metrics);
      });

      room.on('error', (error) => {
        console.error('Room error:', error);
        setConnectionStatus('error');
      });

      // Join the room
      await room.join();
      
      roomRef.current = room;
      setIsInRoom(true);

    } catch (error) {
      console.error('Failed to join room:', error);
      alert(`Failed to join room: ${error.message}`);
    }
  };

  const leaveRoom = async () => {
    if (roomRef.current) {
      await roomRef.current.leave();
      roomRef.current = null;
      setIsInRoom(false);
      setConnectionStatus('disconnected');
      setLastAudioInfo(null);
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getMicLevelWidth = () => {
    return `${Math.min(100, micLevel * 1000)}%`;
  };

  const getCompressionColor = () => {
    const ratio = metrics.compressionRatio || 0;
    if (ratio > 0.9) return 'text-green-600';
    if (ratio > 0.7) return 'text-blue-600';
    return 'text-gray-600';
  };

  const getQualityColor = (value) => {
    if (value > 0.9) return 'text-green-600';
    if (value > 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <main className="flex flex-col items-center gap-6 p-8 max-w-4xl mx-auto">
      <header className="text-center">
        <h1 className="text-3xl font-bold mb-2">Voice Assistant</h1>
        <p className="text-gray-600">Real-time voice with Opus compression</p>
      </header>

      {/* Room Controls */}
      <section className="w-full max-w-md space-y-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value)}
            placeholder="Room ID"
            className="flex-1 px-4 py-2 border rounded-lg"
            disabled={isInRoom}
          />
          <button
            onClick={isInRoom ? leaveRoom : joinRoom}
            className={`px-6 py-2 rounded-lg font-semibold text-white transition-colors ${
              isInRoom ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isInRoom ? 'Leave' : 'Join'} Room
          </button>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
          <span className="text-sm">Status: {connectionStatus}</span>
          {isInRoom && <span className="text-sm text-gray-500">• User: {userId}</span>}
        </div>

        {/* Mic Level Indicator */}
        {isInRoom && (
          <>
            <div className="space-y-1">
              <label className="text-sm text-gray-600">Microphone Level</label>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-green-500 transition-all duration-100"
                  style={{ width: getMicLevelWidth() }}
                />
              </div>
            </div>
            
            {/* VAD Threshold Control */}
            <div className="space-y-1">
              <label className="text-sm text-gray-600">
                Voice Detection Sensitivity (lower = more sensitive)
              </label>
              <input
                type="range"
                min="0.001"
                max="0.05"
                step="0.001"
                defaultValue="0.01"
                onChange={(e) => {
                  if (captureRef.current) {
                    captureRef.current.energyThreshold = parseFloat(e.target.value);
                    console.log('VAD threshold:', e.target.value);
                  }
                }}
                className="w-full"
              />
            </div>
          </>
        )}

        {/* Opus Compression Status */}
        {isInRoom && lastAudioInfo && (
          <div className="p-3 bg-gray-50 rounded-lg space-y-2">
            <h4 className="font-medium text-sm">Audio Compression</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Status:</span>
                <span className={`ml-2 font-medium ${lastAudioInfo.isOpus ? 'text-green-600' : 'text-gray-600'}`}>
                  {lastAudioInfo.isOpus ? 'Opus Active' : 'Raw PCM'}
                </span>
              </div>
              {lastAudioInfo.isOpus && (
                <div>
                  <span className="text-gray-500">Compression:</span>
                  <span className={`ml-2 font-medium ${getCompressionColor()}`}>
                    {((metrics.compressionRatio || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Conversation Display */}
      <section className="w-full max-w-xl space-y-4">
        <div className="space-y-2">
          <h3 className="font-medium">You said:</h3>
          <div className="min-h-[3rem] p-3 bg-gray-50 rounded-lg">
            {transcript || <span className="text-gray-400">Speak into your microphone...</span>}
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="font-medium">Echo Playback:</h3>
          <div className="min-h-[3rem] p-3 bg-blue-50 rounded-lg">
            {assistantMessage || <span className="text-gray-400">You'll hear your voice echo here...</span>}
          </div>
        </div>

        {/* Audio Quality Metrics */}
        {metrics.audioQuality && metrics.audioQuality.snr > 0 && (
          <div className="p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-sm mb-2">Audio Quality</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">SNR:</span>
                <span className={`ml-2 font-medium ${getQualityColor(metrics.audioQuality.snr / 50)}`}>
                  {metrics.audioQuality.snr.toFixed(1)} dB
                </span>
              </div>
              <div>
                <span className="text-gray-500">Correlation:</span>
                <span className={`ml-2 font-medium ${getQualityColor(metrics.audioQuality.correlation)}`}>
                  {(metrics.audioQuality.correlation * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Detailed Metrics */}
      {isInRoom && Object.keys(metrics).length > 0 && (
        <section className="w-full max-w-xl">
          <details className="bg-gray-50 rounded-lg p-4">
            <summary className="cursor-pointer font-medium">Performance Metrics</summary>
            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div>Capture → Send: {metrics.captureToSendLatency?.toFixed(1) || 0}ms</div>
              <div>RTT: {metrics.rtt?.toFixed(1) || 0}ms</div>
              <div>Packets Sent: {metrics.packetsent || 0}</div>
              <div>Packets Received: {metrics.packetReceived || 0}</div>
              <div>Bytes Sent: {((metrics.bytesent || 0) / 1024).toFixed(1)}KB</div>
              <div>Bytes Received: {((metrics.byteReceived || 0) / 1024).toFixed(1)}KB</div>
              <div>Decoder Ready: {metrics.decoderReady ? '✅' : '❌'}</div>
              <div>Bandwidth Saved: {((metrics.compressionRatio || 0) * 100).toFixed(1)}%</div>
            </div>
          </details>
        </section>
      )}

      {/* Instructions */}
      <footer className="text-center text-sm text-gray-600 max-w-lg">
        <p className="mb-2">
          Click "Join Room" and allow microphone access. <strong>Use headphones to prevent echo feedback.</strong>
          Audio is only sent when voice is detected (VAD enabled).
        </p>
        <p className="text-xs">
          Opus compression achieving ~{((metrics.compressionRatio || 0.9) * 100).toFixed(0)}% bandwidth reduction.
          Adjust energyThreshold in AudioCapture if it's too sensitive.
        </p>
      </footer>
    </main>
  );
}