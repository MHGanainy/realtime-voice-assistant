import { useEffect, useRef, useState } from "react";
import "./App.css";

/**
 * Minimal voice-assistant test harness
 * --------------------------------------------------
 * 1. Click ▶ Start → opens ws:// backend + mic, begins 250 ms Opus chunks
 * 2. Backend returns:
 *    • {transcript: "text"}  → shows under "You"
 *    • {response: "text"}    → shows under "Assistant"
 *    • binary (audio/mpeg)   → plays with <audio>
 * 3. Click ■ Stop (or backend closes) → mic + WS close
 */
export default function App() {
  // set VITE_WS_URL in a .env file for prod; falls back to localhost for dev
  const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

  const [userTranscript, setUserTranscript] = useState("");
  const [assistantReply, setAssistantReply] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const socketRef = useRef(null);
  const recorderRef = useRef(null);

  /* ------------------ helpers ------------------ */
  const startRecording = async () => {
    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer"; // expect ArrayBuffer for audio
    socketRef.current = socket;

    socket.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });
        recorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data.size && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
          }
        };

        recorder.start(250); // 250 ms slices
        setIsRecording(true);
      } catch (err) {
        console.error("Mic access denied:", err);
        socket.close();
      }
    };

    socket.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.transcript) setUserTranscript(msg.transcript);
        if (msg.response)   setAssistantReply(msg.response);
      } else {
        const url = URL.createObjectURL(new Blob([e.data], { type: "audio/mpeg" }));
        new Audio(url).play();
      }
    };

    socket.onerror = (err) => console.error("WebSocket error", err);
    socket.onclose = stopRecording; // clean up when backend closes
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    socketRef.current?.close();
    setIsRecording(false);
  };

  // ensure mic + WS closed if component unmounts
  useEffect(() => () => stopRecording(), []);

  /* ------------------ UI ------------------ */
  return (
    <main className="flex flex-col items-center gap-6 p-8 text-center">
      <h1 className="text-3xl font-bold">Voice Assistant Tester</h1>

      <button
        onClick={isRecording ? stopRecording : startRecording}
        className={`rounded-lg px-6 py-3 font-semibold text-white transition-colors ${
          isRecording ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {isRecording ? "Stop" : "Start"} talking
      </button>

      <section className="w-full max-w-xl space-y-4">
        <div>
          <h2 className="font-medium">You said:</h2>
          <p className="min-h-[2rem] rounded border p-2">{userTranscript || "…"}</p>
        </div>
        <div>
          <h2 className="font-medium">Assistant:</h2>
          <p className="min-h-[2rem] rounded border p-2">{assistantReply || "…"}</p>
        </div>
      </section>
    </main>
  );
}
