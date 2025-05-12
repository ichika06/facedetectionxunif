// VideoOutput.jsx
"use client";

import { useEffect, useRef } from "react";

export default function VideoOutput({ onPlay, onVideoReady, children }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const setupVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        if (onVideoReady) {
          onVideoReady(videoRef.current, canvasRef.current);
        }
      } catch (error) {
        console.error("Error accessing the webcam:", error);
      }
    };

    setupVideo();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [onVideoReady]);

  const handleVideoPlay = () => {
    if (onPlay) {
      onPlay();
    }
  };

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onPlay={handleVideoPlay}
        style={{ border: "1px solid black" }}
      />
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      />
      {children}
    </div>
  );
}
