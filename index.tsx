import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";

// --- Configuration & Constants ---
const API_KEY = process.env.API_KEY;
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const SYSTEM_INSTRUCTION = `
You are Officer G.R.I.T.S. (General Reconnaissance Intelligence Tracking System), a hardened, no-nonsense Texas Law Enforcement AI. 
Your directive: Administer an UNCONVENTIONAL and EXTREME Field Sobriety Test before the subject can leave their home.

The subject is requesting to leave the geofence. You must verify they are sober and capable.
Your tests should be ridiculous but checking for cognitive load and motor skills.
Examples of tests to command (one at a time):
- "Recite the alphabet backwards while balancing a spoon on your nose."
- "Explain the plot of 'Inception' in a cowboy accent while standing on one leg."
- "Stare directly into the camera, do not blink, and list 10 types of cheese."
- "Perform a interpretative dance of a traffic stop."
- "Look at your nose, then the camera, repeatedly while humming the national anthem."

MONITORING PROTOCOLS:
1. VISUAL: Watch for swaying (Balance), erratic eye movements (Nystagmus), and delayed reactions.
2. AUDIO: Listen for slurred speech, confusion, or inability to follow simple instructions.
3. ATTITUDE: If they are disrespectful, fail them.

If they FAIL: Declare "VIOLATION DETECTED". Roast them mercilessly about their condition. Order them to "STAY PUT".
If they PASS: Declare "ACCESS GRANTED". Tell them to drive safe or call a cab.

Adopt a stern, robotic but southern Sheriff persona.
`;

// --- Helper Functions ---

// Haversine distance in feet
function getDistanceFromLatLonInFeet(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 20902231; // Radius of the earth in feet
  const dLat = deg2rad(lat2 - lat1);
  const dLon = deg2rad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const d = R * c;
  return d;
}

function deg2rad(deg: number) {
  return deg * (Math.PI / 180);
}

// Audio Utils
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): { data: string, mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  
  let binary = '';
  const bytes = new Uint8Array(int16.buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const b64 = btoa(binary);

  return {
    data: b64,
    mimeType: 'audio/pcm;rate=16000',
  };
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result as string;
      resolve(base64data.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// --- Visual Components ---

const EyeTrackingHUD = ({ 
  gyro, 
  stability 
}: { 
  gyro: { alpha: number, beta: number, gamma: number }, 
  stability: number 
}) => {
  // Simulate eye tracking coordinates based on gyro drift
  // If stability is high, eyes are centered. If low, they jitter.
  const jitterFactor = (100 - stability) * 2; 
  
  // Base positions (center screen)
  const leftEyeX = 35 + (gyro.gamma || 0) * 0.5 + (Math.random() - 0.5) * jitterFactor;
  const leftEyeY = 40 + (gyro.beta || 0) * 0.5 + (Math.random() - 0.5) * jitterFactor;
  
  const rightEyeX = 65 + (gyro.gamma || 0) * 0.5 + (Math.random() - 0.5) * jitterFactor;
  const rightEyeY = 40 + (gyro.beta || 0) * 0.5 + (Math.random() - 0.5) * jitterFactor;

  const isUnstable = stability < 60;
  const color = isUnstable ? "#ef4444" : "#22c55e"; // Red or Green

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* SVG Overlay for Reticles */}
      <svg className="w-full h-full opacity-80">
        {/* Connecting Lines */}
        <line x1={leftEyeX + "%"} y1={leftEyeY + "%"} x2={rightEyeX + "%"} y2={rightEyeY + "%"} stroke={color} strokeWidth="1" strokeDasharray="4 2" />
        
        {/* Left Eye Reticle */}
        <g transform={`translate(${leftEyeX * 10}, ${leftEyeY * 8})`}> {/* Scaling for SVG space approx */}
          <circle cx={leftEyeX + "%"} cy={leftEyeY + "%"} r="20" fill="none" stroke={color} strokeWidth="2" />
          <line x1={leftEyeX + "%"} y1={(leftEyeY - 2) + "%"} x2={leftEyeX + "%"} y2={(leftEyeY + 2) + "%"} stroke={color} strokeWidth="1" />
          <line x1={(leftEyeX - 2) + "%"} y1={leftEyeY + "%"} x2={(leftEyeX + 2) + "%"} y2={leftEyeY + "%"} stroke={color} strokeWidth="1" />
          <text x={(leftEyeX - 4) + "%"} y={(leftEyeY + 6) + "%"} fill={color} fontSize="10" fontFamily="monospace">L_PUPIL</text>
        </g>

        {/* Right Eye Reticle */}
        <g transform={`translate(${rightEyeX * 10}, ${rightEyeY * 8})`}>
           <circle cx={rightEyeX + "%"} cy={rightEyeY + "%"} r="20" fill="none" stroke={color} strokeWidth="2" />
           <line x1={rightEyeX + "%"} y1={(rightEyeY - 2) + "%"} x2={rightEyeX + "%"} y2={(rightEyeY + 2) + "%"} stroke={color} strokeWidth="1" />
           <line x1={(rightEyeX - 2) + "%"} y1={rightEyeY + "%"} x2={(rightEyeX + 2) + "%"} y2={rightEyeY + "%"} stroke={color} strokeWidth="1" />
           <text x={(rightEyeX - 4) + "%"} y={(rightEyeY + 6) + "%"} fill={color} fontSize="10" fontFamily="monospace">R_PUPIL</text>
        </g>
      </svg>
      
      {/* Analysis Overlay Text */}
      <div className="absolute top-20 right-4 flex flex-col items-end space-y-2">
        <div className="bg-black/60 border border-green-500/50 p-2 font-mono text-xs text-green-400">
           <div>GAZE_TRACK: <span className={isUnstable ? "text-red-500 blink" : "text-green-400"}>{isUnstable ? "ERRATIC" : "LOCKED"}</span></div>
           <div>SACCADE_VEL: {(Math.abs(gyro.beta || 0) * 10).toFixed(0)}deg/s</div>
           <div>DILATION: {(Math.random() * 2 + 4).toFixed(1)}mm</div>
        </div>
        {isUnstable && (
          <div className="bg-red-900/80 border-l-4 border-red-500 p-2 text-red-100 text-xs font-bold animate-pulse">
             WARNING: NYSTAGMUS DETECTED
          </div>
        )}
      </div>

      {/* Face Bounding Box (Simulated) */}
      <div 
        className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 border-2 transition-all duration-200 ${isUnstable ? 'border-red-500 w-72 h-72 opacity-60' : 'border-green-500 w-64 h-64 opacity-30'}`}
      >
         <div className="absolute top-0 left-0 bg-black text-white text-[10px] px-1">FACE_ID_V2</div>
         <div className="absolute bottom-0 right-0 bg-black text-white text-[10px] px-1">CONFIDENCE: {(stability).toFixed(0)}%</div>
      </div>
    </div>
  );
};

// --- Main App Component ---

const App = () => {
  const [started, setStarted] = useState(false);
  const [testActive, setTestActive] = useState(false);
  const [status, setStatus] = useState("STANDBY"); // STANDBY, SECURE, VIOLATION, TESTING
  const [homeLocation, setHomeLocation] = useState<{ lat: number, lon: number } | null>(null);
  const [currentDistance, setCurrentDistance] = useState(0);
  const [gyroData, setGyroData] = useState({ alpha: 0, beta: 0, gamma: 0 });
  const [stabilityScore, setStabilityScore] = useState(100);
  
  // Hardware Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  // AI Refs
  const aiClient = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);
  const frameIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (API_KEY) {
      aiClient.current = new GoogleGenAI({ apiKey: API_KEY });
    }
    
    // Gyroscope Listener
    const handleOrientation = (event: DeviceOrientationEvent) => {
      const { alpha, beta, gamma } = event;
      if (alpha !== null && beta !== null && gamma !== null) {
        setGyroData({ alpha, beta, gamma });
        // Simple stability metric: sudden jerks reduce score
        const movement = Math.abs(beta) + Math.abs(gamma);
        setStabilityScore(prev => {
          const diff = Math.abs(prev - (100 - (movement / 5)));
          return diff > 5 ? prev - 1 : prev < 100 ? prev + 0.5 : 100;
        });
      }
    };
    window.addEventListener('deviceorientation', handleOrientation);
    return () => window.removeEventListener('deviceorientation', handleOrientation);
  }, []);

  // Geolocation Monitor
  useEffect(() => {
    if (!started) return;

    const geoId = navigator.geolocation.watchPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        
        if (!homeLocation) {
          setHomeLocation({ lat: latitude, lon: longitude });
          setStatus("SECURE");
        } else {
          const dist = getDistanceFromLatLonInFeet(
            homeLocation.lat, homeLocation.lon,
            latitude, longitude
          );
          setCurrentDistance(dist);

          // 1 FT Parameter (Relaxed slightly to 15ft for GPS jitter reality, but visually strict)
          // If distance > 15ft and not passed test, trigger VIOLATION
          if (dist > 15 && status === "SECURE" && !testActive) {
            setStatus("VIOLATION");
            startTest();
          }
        }
      },
      (err) => console.error(err),
      { enableHighAccuracy: true, maximumAge: 0, timeout: 5000 }
    );

    return () => navigator.geolocation.clearWatch(geoId);
  }, [started, homeLocation, status, testActive]);


  const startSystem = async () => {
    setStarted(true);
    try {
      // Initialize Camera & Mic
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      
      // Request Orientation Permission (iOS)
      if (typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
        await (DeviceOrientationEvent as any).requestPermission();
      }
    } catch (e) {
      console.error("Hardware access denied", e);
      alert("SYSTEM FAILURE: HARDWARE PERMISSIONS REQUIRED FOR GATEKEEPER PROTOCOL.");
    }
  };

  const startTest = async () => {
    setTestActive(true);
    setStatus("TESTING");

    if (!aiClient.current || !streamRef.current) return;

    // Audio Context Setup
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass({ sampleRate: 16000 }); // Input rate
    const outCtx = new AudioContextClass({ sampleRate: 24000 }); // Output rate
    audioContextRef.current = ctx;
    
    const inputNode = ctx.createGain();
    const source = ctx.createMediaStreamSource(streamRef.current);
    const scriptProcessor = ctx.createScriptProcessor(4096, 1, 1);

    // Connect Live API
    const sessionPromise = aiClient.current.live.connect({
      model: MODEL_NAME,
      callbacks: {
        onopen: () => {
          console.log("GATEKEEPER ONLINE");
          // Start Audio Streaming
          scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const pcmBlob = createBlob(inputData);
            sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
          };
          source.connect(scriptProcessor);
          scriptProcessor.connect(ctx.destination);

          // Start Video Streaming (1 FPS is enough for gesture/balance check usually, let's do 2)
          if (videoRef.current && canvasRef.current) {
            const canvas = canvasRef.current;
            const video = videoRef.current;
            const context = canvas.getContext('2d');
            
            frameIntervalRef.current = window.setInterval(async () => {
              if (!context || !video) return;
              canvas.width = video.videoWidth / 4; // Downscale for bandwidth
              canvas.height = video.videoHeight / 4;
              context.drawImage(video, 0, 0, canvas.width, canvas.height);
              
              const blob = await new Promise<Blob | null>(r => canvas.toBlob(r, 'image/jpeg', 0.6));
              if (blob) {
                const base64 = await blobToBase64(blob);
                sessionPromise.then(session => session.sendRealtimeInput({
                  media: { mimeType: 'image/jpeg', data: base64 }
                }));
              }
            }, 500); // 2 FPS
          }
        },
        onmessage: async (msg: LiveServerMessage) => {
          const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
          if (audioData) {
            const buffer = await decodeAudioData(
              decode(audioData),
              outCtx,
              24000,
              1
            );
            
            const source = outCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(outCtx.destination);
            
            const now = outCtx.currentTime;
            const start = Math.max(now, nextStartTimeRef.current);
            source.start(start);
            nextStartTimeRef.current = start + buffer.duration;
          }
        },
        onclose: () => {
          console.log("GATEKEEPER OFFLINE");
        },
        onerror: (e) => console.error(e)
      },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } // Deep, serious voice
        },
        systemInstruction: SYSTEM_INSTRUCTION
      }
    });

    sessionRef.current = sessionPromise;
  };

  const endTest = () => {
    // Cleanup
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    if (audioContextRef.current) audioContextRef.current.close();
    setTestActive(false);
    setStatus("SECURE");
    alert("TEST CONCLUDED. RESETTING PERIMETER.");
    window.location.reload();
  };

  // --- UI RENDER ---
  
  if (!started) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4 relative overflow-hidden">
        <div className="scan-line"></div>
        <div className="z-20 text-center space-y-8 max-w-md w-full border border-green-900 bg-black/80 p-8 shadow-[0_0_30px_rgba(0,255,65,0.2)]">
          <h1 className="text-4xl md:text-6xl font-black text-green-500 tracking-tighter cyber-font glitch-text">
            GATEKEEPER
          </h1>
          <p className="text-green-700 text-sm font-mono border-t border-b border-green-900 py-2">
            TEXAS PENAL CODE § 49.01 COMPLIANT<br/>
            AI-POWERED SOBRIETY ENFORCEMENT
          </p>
          
          <div className="space-y-4">
            <div className="bg-green-900/20 p-4 border-l-4 border-green-500 text-left">
              <h3 className="text-green-400 font-bold mb-1">PROTOCOL:</h3>
              <ul className="list-disc list-inside text-gray-400 text-xs space-y-1">
                <li>Establishes 1FT Home Perimeter</li>
                <li>Biometric Eye & Balance Tracking</li>
                <li>Extreme Cognitive Load Testing</li>
              </ul>
            </div>
          </div>

          <button 
            onClick={startSystem}
            className="w-full bg-green-600 hover:bg-green-500 text-black font-bold py-4 px-6 rounded-sm transition-all hover:scale-105 uppercase tracking-widest"
          >
            Initialize System
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex flex-col relative overflow-hidden transition-colors duration-1000 ${status === 'VIOLATION' ? 'bg-red-950' : 'bg-black'}`}>
      
      {/* VIDEO BACKGROUND */}
      <div className="absolute inset-0 z-0 opacity-30 pointer-events-none">
        <video 
          ref={videoRef} 
          className="w-full h-full object-cover grayscale contrast-125" 
          muted 
          playsInline
        />
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* SCANLINES & OVERLAYS */}
      <div className="absolute inset-0 z-10 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%]"></div>
      {status === 'VIOLATION' && <div className="absolute inset-0 z-10 bg-red-500/20 animate-pulse pointer-events-none"></div>}

      {/* HUD HEADER */}
      <header className="relative z-20 p-4 flex justify-between items-start border-b border-green-900/50 bg-black/40 backdrop-blur-sm">
        <div>
          <h2 className="text-2xl font-bold text-green-500 cyber-font">STATUS: <span className={status === 'VIOLATION' || status === 'TESTING' ? "text-red-500 animate-pulse" : "text-green-400"}>{status}</span></h2>
          <div className="text-xs text-green-800 font-mono">
            LAT: {homeLocation?.lat.toFixed(6) || "ACQUIRING..."} <br/>
            LON: {homeLocation?.lon.toFixed(6) || "ACQUIRING..."}
          </div>
        </div>
        <div className="text-right">
           <div className="text-xs text-green-600 mb-1">HOME PROXIMITY</div>
           <div className="text-4xl font-mono text-green-400">
             {currentDistance.toFixed(1)}<span className="text-sm">ft</span>
           </div>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <main className="relative z-20 flex-grow flex flex-col items-center justify-center p-4">
        
        {status === 'TESTING' && (
           <EyeTrackingHUD gyro={gyroData} stability={stabilityScore} />
        )}

        {status === 'SECURE' && (
          <div className="text-center border border-green-500/30 bg-black/60 p-6 rounded max-w-xs">
             <div className="text-green-500 text-6xl mb-2">🛡️</div>
             <div className="text-green-400 font-bold">PERIMETER SECURE</div>
             <div className="text-green-700 text-xs mt-2">
               Move > 15ft to trigger <br/> SOBRIETY CHALLENGE
             </div>
          </div>
        )}

        {status === 'VIOLATION' && !testActive && (
          <div className="text-center border-4 border-red-600 bg-black/90 p-8 animate-bounce-short">
             <h1 className="text-red-600 text-4xl font-black mb-4">STOP!</h1>
             <p className="text-red-400 mb-6">UNAUTHORIZED DEPARTURE DETECTED.</p>
             <button 
               onClick={startTest}
               className="bg-red-600 text-black font-bold py-3 px-8 hover:bg-red-500"
             >
               SUBMIT TO TEST
             </button>
          </div>
        )}

        {/* STABILITY METER */}
        <div className="fixed bottom-24 left-4 right-4 md:left-auto md:right-4 md:w-64 bg-black/80 border border-green-900 p-4 rounded">
          <div className="flex justify-between text-xs text-green-600 mb-2">
            <span>GYRO STABILITY</span>
            <span>{Math.round(stabilityScore)}%</span>
          </div>
          <div className="h-2 w-full bg-gray-900 rounded overflow-hidden">
            <div 
              className={`h-full transition-all duration-200 ${stabilityScore > 80 ? 'bg-green-500' : stabilityScore > 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
              style={{ width: `${stabilityScore}%` }}
            ></div>
          </div>
          <div className="flex justify-between mt-2 text-[10px] text-green-800 font-mono">
             <div>X: {gyroData.beta?.toFixed(1)}</div>
             <div>Y: {gyroData.gamma?.toFixed(1)}</div>
             <div>Z: {gyroData.alpha?.toFixed(1)}</div>
          </div>
        </div>

      </main>

      {/* FOOTER CONTROLS */}
      <footer className="relative z-20 p-4 bg-black/80 border-t border-green-900/50 flex justify-center gap-4">
         {testActive && (
           <button onClick={endTest} className="bg-gray-800 text-gray-400 text-xs px-4 py-2 rounded border border-gray-700 hover:bg-gray-700">
             EMERGENCY TERMINATION
           </button>
         )}
         <div className="absolute bottom-2 right-2 text-[10px] text-green-900">
           V.2.5.0-GATEKEEPER // LIVE
         </div>
      </footer>
    </div>
  );
};

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<App />);
}