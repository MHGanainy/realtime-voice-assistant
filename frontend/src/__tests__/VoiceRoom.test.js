import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VoiceRoom } from '../voice/VoiceRoom';
import { AudioCapture } from '../audio/AudioCapture';
import { AudioPlayer } from '../audio/AudioPlayer';

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = WebSocket.CONNECTING;
    this.binaryType = 'arraybuffer';
    this.sentData = null;
    this.onopen = null;
    this.onclose = null;
    this.onmessage = null;
    this.onerror = null;
    
    // Simulate connection opening after a short delay
    setTimeout(() => {
      if (this.readyState !== WebSocket.CLOSED) {
        this.readyState = WebSocket.OPEN;
        this.onopen?.();
      }
    }, 10);
  }

  send(data) {
    if (this.readyState === WebSocket.OPEN) {
      this.sentData = data;
    }
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.();
  }
}

global.WebSocket = MockWebSocket;
global.WebSocket.CONNECTING = 0;
global.WebSocket.OPEN = 1;
global.WebSocket.CLOSING = 2;
global.WebSocket.CLOSED = 3;

describe('VoiceRoom', () => {
  let room;
  let mockCapture;
  let mockPlayer;

  beforeEach(() => {
    vi.useFakeTimers();
    
    mockCapture = new AudioCapture();
    mockPlayer = new AudioPlayer();
    
    // Mock the audio components
    vi.spyOn(mockCapture, 'initialize').mockResolvedValue();
    vi.spyOn(mockCapture, 'start').mockResolvedValue();
    vi.spyOn(mockCapture, 'stop').mockResolvedValue();
    vi.spyOn(mockPlayer, 'initialize').mockResolvedValue();
    vi.spyOn(mockPlayer, 'play').mockResolvedValue();

    room = new VoiceRoom({
      roomId: 'test-room',
      userId: 'test-user',
      capture: mockCapture,
      player: mockPlayer
    });
  });

  afterEach(async () => {
    // Clean up any active connections
    if (room && room.socket) {
      room.shouldReconnect = false;
      await room.leave();
    }
    vi.clearAllMocks();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  describe('Room Connection', () => {
    it('should connect to WebSocket with room ID', async () => {
      const joinPromise = room.join();
      
      // Fast-forward timer to open WebSocket
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;
      
      expect(room.socket).toBeDefined();
      expect(room.socket.url).toContain('/ws/room/test-room');
      expect(room.socket.binaryType).toBe('arraybuffer');
    });

    it('should send join message on connection', async () => {
      const joinPromise = room.join();
      
      // Fast-forward to open connection
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      const sentData = JSON.parse(room.socket.sentData);
      expect(sentData).toEqual({
        type: 'join',
        userId: 'test-user'
      });
    });

    it('should initialize audio capture after joining', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      expect(mockCapture.initialize).toHaveBeenCalled();
      expect(mockCapture.start).toHaveBeenCalled();
    });
  });

  describe('Audio Streaming', () => {
    it('should send audio chunks via WebSocket', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      // Simulate audio chunk from capture
      const audioChunk = new Uint8Array([1, 2, 3, 4]);
      mockCapture.emit('audioChunk', { data: audioChunk, encoded: true });

      expect(room.socket.sentData).toEqual(audioChunk);
    });

    it('should handle received audio and play it', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      // Simulate receiving audio from server
      const audioData = new ArrayBuffer(8);
      room.socket.onmessage({ data: audioData });

      expect(mockPlayer.play).toHaveBeenCalledWith(audioData);
    });

    it('should handle JSON messages for events', async () => {
      const onParticipantJoined = vi.fn();
      room.on('participantJoined', onParticipantJoined);

      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      // Simulate participant joined message
      const message = JSON.stringify({
        type: 'participant_joined',
        userId: 'other-user'
      });
      room.socket.onmessage({ data: message });

      expect(onParticipantJoined).toHaveBeenCalledWith({ userId: 'other-user' });
    });
  });

  describe('Voice Activity', () => {
    it('should forward VAD events to server', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      // Simulate voice start
      mockCapture.emit('voiceStart');

      const sentData = JSON.parse(room.socket.sentData);
      expect(sentData).toEqual({
        type: 'voice_activity',
        activity: 'start'
      });
    });

    it('should handle interrupt when speaking during TTS', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;

      // Simulate TTS playing
      room.isTTSPlaying = true;

      // User starts speaking
      mockCapture.emit('voiceStart');

      const sentData = JSON.parse(room.socket.sentData);
      expect(sentData).toEqual({
        type: 'interrupt'
      });
    });
  });

  describe('Connection Management', () => {
    it('should handle reconnection on disconnect', async () => {
      vi.useRealTimers(); // Use real timers for this test
      
      await room.join();
      const firstSocket = room.socket;

      // Simulate disconnect
      room.socket.onclose();

      // Wait for reconnection
      await new Promise(resolve => setTimeout(resolve, 1100));

      expect(room.socket).not.toBe(firstSocket);
      expect(room.socket.readyState).toBe(WebSocket.OPEN);
      
      vi.useFakeTimers(); // Switch back to fake timers
    });

    it('should clean up resources on leave', async () => {
      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;
      
      await room.leave();

      expect(mockCapture.stop).toHaveBeenCalled();
      expect(room.socket).toBe(null);
      expect(room.isConnected).toBe(false);
    });
  });

  describe('Metrics', () => {
    it('should track audio latency metrics', async () => {
      vi.useRealTimers(); // Use real timers for performance.now()
      
      const joinPromise = room.join();
      await new Promise(resolve => setTimeout(resolve, 20));
      await joinPromise;

      // Send audio with timestamp using performance.now()
      const timestamp = performance.now();
      const audioChunk = { 
        data: new Uint8Array([1, 2, 3]), 
        timestamp: timestamp,
        encoded: true 
      };
      
      // Add small delay to ensure positive latency
      await new Promise(resolve => setTimeout(resolve, 5));
      
      mockCapture.emit('audioChunk', audioChunk);

      expect(room.metrics.captureToSendLatency).toBeGreaterThan(0);
      expect(room.metrics.captureToSendLatency).toBeLessThan(100);
      
      vi.useFakeTimers(); // Switch back
    });

    it('should emit metric events', async () => {
      const onMetrics = vi.fn();
      room.on('metrics', onMetrics);

      const joinPromise = room.join();
      await vi.advanceTimersByTimeAsync(20);
      await joinPromise;
      
      // Trigger metric collection
      room.collectMetrics();

      expect(onMetrics).toHaveBeenCalledWith(
        expect.objectContaining({
          captureToSendLatency: expect.any(Number),
          packetsent: expect.any(Number),
          packetReceived: expect.any(Number)
        })
      );
    });
  });
});