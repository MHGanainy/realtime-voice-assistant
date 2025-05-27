import { useEffect, useRef, useState } from "react";
import "./App.css";

/**
 * Real-time voice-assistant test harness
 * -------------------------------------
 * 1. ▶ Start → opens ws:// backend + mic, begins 250 ms Opus chunks
 * 2. Backend returns:
 *      • {command:"pause"|"resume"}            → flow-control
 *      • {transcript:"…", final:false}         → live caption (optional)
 *      • {transcript:"…", final:true,
 *         response:"…"}                        → finished utterance
 *      • binary (audio/mpeg)                   → ElevenLabs MP3 stream
 * 3. ■ Stop (or backend closes) → mic + WS close
 */
export default function App() {
  const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

  const [userTranscript, setUserTranscript] = useState("");
  const [assistantReply, setAssistantReply] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const socketRef             = useRef(null);
  const recorderRef           = useRef(null);
  const sendAudioRef          = useRef(true);      // backend flow-flag
  const expectNewAudioRef     = useRef(false);     // set on "pause"

  /* ---- media-pipeline plumbing ---------------------------------------- */
  const mediaSrcRef           = useRef(null);      // MediaSource
  const sourceBufRef          = useRef(null);      // SourceBuffer (audio/mpeg)
  const pendingChunksRef      = useRef([]);        // Uint8Array[] waiting
  const audioRef              = useRef(null);      // <audio> element

  /* ---- helpers -------------------------------------------------------- */
  function teardownMediaSource() {
    audioRef.current?.pause();
    if (audioRef.current?.src?.startsWith("blob:")) {
      URL.revokeObjectURL(audioRef.current.src);
    }
    mediaSrcRef.current = null;
    sourceBufRef.current = null;
    pendingChunksRef.current = [];
    audioRef.current = null;
  }

  function initMediaSource() {
    if (mediaSrcRef.current) return;               // already open

    mediaSrcRef.current = new MediaSource();
    const url = URL.createObjectURL(mediaSrcRef.current);
    audioRef.current = new Audio(url);

    audioRef.current.play().catch(() => {
      /* user gesture may be required – ignore here, we retry later */
    });

    mediaSrcRef.current.addEventListener("sourceopen", () => {
      sourceBufRef.current =
        mediaSrcRef.current.addSourceBuffer("audio/mpeg");
      sourceBufRef.current.mode = "sequence";      // back-to-back MP3

      sourceBufRef.current.addEventListener("updateend", flushPending);
      flushPending();                              // flush early chunks
    });
  }

  function flushPending() {
    const sb = sourceBufRef.current;
    if (!sb || sb.updating) return;

    const chunk = pendingChunksRef.current.shift();
    if (!chunk) return;

    sb.appendBuffer(chunk);

    /* once the chunk is committed, (re)start playback if needed */
    const kick = () => {
      if (audioRef.current?.paused) {
        audioRef.current.play().catch(() => {});
      }
      sb.removeEventListener("updateend", kick);
    };
    sb.addEventListener("updateend", kick);
  }

  const startRecording = async () => {
    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });
        recorderRef.current = recorder;
        sendAudioRef.current = true;

        recorder.ondataavailable = async (e) => {
          if (
            !e.data.size ||
            socket.readyState !== WebSocket.OPEN ||
            !sendAudioRef.current
          )
            return;
          const buf = await e.data.arrayBuffer();
          socket.send(buf);
        };

        recorder.start(250);
        setIsRecording(true);
      } catch (err) {
        console.error("Mic access denied:", err);
        socket.close();
      }
    };

    socket.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);

        /* ---- flow-control from backend ----------------------------- */
        if (msg.command === "pause") {
          sendAudioRef.current = false;
          expectNewAudioRef.current = true;        // next binary = new reply
          recorderRef.current?.pause();
          return;
        }
        if (msg.command === "resume") {
          sendAudioRef.current = true;
          recorderRef.current?.resume?.();
          return;
        }

        /* ---- transcript / assistant text --------------------------- */
        if (msg.final) {
          setUserTranscript(msg.transcript);
          setAssistantReply(msg.response ?? msg.transcript ?? "");
        } else if (msg.transcript) {
          setUserTranscript(msg.transcript);
        }
      } else {
        /* ---- streamed MP3 bytes from ElevenLabs -------------------- */
        if (expectNewAudioRef.current) {
          /* first chunk of a brand-new TTS reply – rebuild pipeline   */
          teardownMediaSource();
          expectNewAudioRef.current = false;
        }

        initMediaSource();
        pendingChunksRef.current.push(new Uint8Array(e.data));
        flushPending();
      }
    };

    socket.onerror = (err) => console.error("WebSocket error", err);
    socket.onclose  = stopRecording;
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    socketRef.current?.close();
    setIsRecording(false);
    setUserTranscript("");
    setAssistantReply("");
    teardownMediaSource();
  };

  useEffect(() => () => stopRecording(), []);

  /* ---- UI ----------------------------------------------------------- */
  return (
    <main className="flex flex-col items-center gap-6 p-8 text-center">
      <h1 className="text-3xl font-bold">Voice Assistant Tester</h1>

      <button
        onClick={isRecording ? stopRecording : startRecording}
        className={`rounded-lg px-6 py-3 font-semibold text-white transition-colors ${
          isRecording
            ? "bg-red-600 hover:bg-red-700"
            : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {isRecording ? "Stop" : "Start"} talking
      </button>

      <section className="w-full max-w-xl space-y-4">
        <div>
          <h2 className="font-medium">You said:</h2>
          <p className="min-h-[2rem] rounded border p-2">
            {userTranscript || "…"}
          </p>
        </div>
        <div>
          <h2 className="font-medium">Assistant:</h2>
          <p className="min-h-[2rem] rounded border p-2">
            {assistantReply || "…"}
          </p>
        </div>
      </section>
    </main>
  );
}
