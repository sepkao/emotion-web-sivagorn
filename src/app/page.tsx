"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("กำลังเตรียมระบบ...");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // --- Logic เดิมคงไว้ (OpenCV, Cascade, Model Loading) ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else { setTimeout(waitReady, 50); }
        };
        if ("onRuntimeInitialized" in cv) cv.onRuntimeInitialized = () => waitReady();
        else waitReady();
      };
      script.onerror = () => reject(new Error("โหลด OpenCV ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());
    const cascadePath = "haarcascade_frontalface_default.xml";
    try { cv.FS_unlink(cascadePath); } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    faceCascade.load(cascadePath);
    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    const session = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx", { executionProviders: ["wasm"] });
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    classesRef.current = await clsRes.json();
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("ระบบกำลังทำงาน");
      requestAnimationFrame(loop);
    } catch (e) {
      setStatus("ไม่สามารถเข้าถึงกล้องได้");
    }
  }

  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size; tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);
    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        float[idx++] = imgData[i * 4 + c] / 255;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      const faces = new cv.RectVector();
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, new cv.Size(0,0), new cv.Size(0,0));

      let bestRect: any = null;
      let bestArea = 0;
      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) { bestArea = area; bestRect = r; }
        ctx.strokeStyle = "#10b981"; // Emerald-500
        ctx.lineWidth = 3;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width; faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds: any = {}; feeds[session.inputNames[0]] = input;
        const out = await session.run(feeds);
        const logits = out[session.outputNames[0]].data as Float32Array;
        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;

        setEmotion(classes[maxIdx]);
        setConf(probs[maxIdx]);

        // Draw Label UI on Canvas
        const label = `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(0)}%`;
        ctx.font = "bold 18px sans-serif";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = "#10b981";
        ctx.fillRect(bestRect.x, bestRect.y - 30, textWidth + 12, 30);
        ctx.fillStyle = "white";
        ctx.fillText(label, bestRect.x + 6, bestRect.y - 8);
      }

      src.delete(); gray.delete(); faces.delete();
      requestAnimationFrame(loop);
    } catch (e) { console.error(e); }
  }

  useEffect(() => {
    (async () => {
      try {
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setStatus("พร้อมใช้งาน");
        setIsLoading(false);
      } catch (e) { setStatus("เกิดข้อผิดพลาดในการโหลด"); }
    })();
  }, []);

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 font-sans p-4 md:p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-extrabold tracking-tight lg:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-emerald-500 to-sky-500">
            Emotion AI
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400">
            วิเคราะห์อารมณ์จากใบหน้าแบบ Real-time ด้วย YOLO11
          </p>
        </header>

        {/* Status & Result Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-zinc-900 p-4 rounded-2xl border border-zinc-200 dark:border-zinc-800 shadow-sm">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">สถานะระบบ</p>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-2 h-2 rounded-full ${status.includes("พร้อม") || status.includes("ทำงาน") ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
              <p className="font-semibold">{status}</p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-zinc-900 p-4 rounded-2xl border border-zinc-200 dark:border-zinc-800 shadow-sm col-span-1 md:col-span-2 flex justify-between items-center">
            <div>
              <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">ผลการวิเคราะห์</p>
              <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mt-1">
                {emotion} <span className="text-lg font-normal text-zinc-400">{(conf * 100).toFixed(1)}%</span>
              </p>
            </div>
            <button
              disabled={isLoading}
              onClick={startCamera}
              className="px-6 py-2.5 bg-zinc-900 dark:bg-zinc-100 dark:text-zinc-900 text-white rounded-xl font-bold hover:opacity-90 transition-all disabled:opacity-50 active:scale-95"
            >
              Start Camera
            </button>
          </div>
        </div>

        {/* Video/Canvas Container */}
        <div className="relative group overflow-hidden rounded-3xl border-4 border-white dark:border-zinc-900 shadow-2xl bg-zinc-200 dark:bg-zinc-800 aspect-video flex items-center justify-center">
          <video ref={videoRef} className="hidden" playsInline />
          <canvas ref={canvasRef} className="w-full h-full object-cover" />
          
          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-900/10 backdrop-blur-sm">
              <div className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="font-medium">กำลังเตรียมโมเดล AI...</p>
            </div>
          )}
          
          {!isLoading && status === "พร้อมใช้งาน" && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-transparent transition-colors pointer-events-none">
              <p className="bg-black/50 text-white px-4 py-2 rounded-full text-sm backdrop-blur-md">
                กดปุ่ม Start เพื่อเริ่มใช้งาน
              </p>
            </div>
          )}
        </div>

        {/* Footer info */}
        <footer className="flex flex-col md:flex-row justify-between items-center text-xs text-zinc-400 pt-4 border-t border-zinc-200 dark:border-zinc-800">
          <p>Powered by OpenCV.js & ONNX Runtime Web</p>
          <p>YOLO11 Classification Model</p>
        </footer>
      </div>
    </main>
  );
}