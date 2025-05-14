"use client";

import * as tf from "@tensorflow/tfjs";
import React, { useRef, useEffect, useState, useCallback } from "react";
import * as faceapi from "face-api.js";
import Webcam from "react-webcam";
import { motion, AnimatePresence } from "framer-motion";

function FaceDetection() {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const modelRef = useRef(null);

  const [detections, setDetections] = useState([]);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [clothingPredictions, setClothingPredictions] = useState([]);
  const [classNames, setClassNames] = useState([]);
  const [isModelsLoaded, setIsModelsLoaded] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isTFReady, setIsTFReady] = useState(false);
  const [detectedClothesAboveThreshold, setDetectedClothesAboveThreshold] = useState([]);

  const predictionThreshold = 0.4;

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = "/models/faceapi";
      const TF_MODEL_URL = "/models/clothing/model.json";
      const METADATA_URL = "/models/clothing/metadata.json";

      try {
        await tf.setBackend('webgl'); // Or 'cpu' if you prefer

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        ]);
        console.log("Face API models loaded");

        const model = await tf.loadLayersModel(TF_MODEL_URL);
        modelRef.current = model;
        console.log("TensorFlow model loaded", model);
        setIsTFReady(true);

        const metadataResponse = await fetch(METADATA_URL);
        const metadata = await metadataResponse.json();
        setClassNames(metadata.labels);
        console.log("Metadata loaded", metadata);

        setIsModelsLoaded(true);
      } catch (error) {
        console.error("Error loading models:", error);
      }
    };

    loadModels();
  }, []);

  const runClothingPrediction = useCallback(async () => {
    if (!modelRef.current || !videoRef.current || !isVideoReady || !isTFReady) {
      return;
    }

    try {
      const videoElement = videoRef.current.video || videoRef.current;

      if (!videoElement || videoElement.readyState !== 4 || videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        console.log("Video element not fully ready for prediction");
        return;
      }

      const imageTensor = tf.browser.fromPixels(videoElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

      const predictions = await modelRef.current.predict(imageTensor).data();

      const processedPredictions = [];
      if (predictions && classNames.length > 0) {
        for (let i = 0; i < classNames.length; i++) {
          const probability = predictions[i];
          if (probability > predictionThreshold) {
            const x = 10; // Fixed x-coordinate for bottom left
            const y = videoElement.videoHeight - 110; // Fixed y-coordinate for bottom left
            const width = 100;
            const height = 100;

            processedPredictions.push({
              bbox: [x / videoElement.videoWidth, y / videoElement.videoHeight, width / videoElement.videoWidth, height / videoElement.videoHeight], // Normalized
              className: classNames[i],
              probability: probability,
            });
          }
        }
      }

      const scaledPredictions = processedPredictions.map(prediction => ({
        ...prediction,
        scaledBBox: [
          prediction.bbox[0] * videoElement.videoWidth,
          prediction.bbox[1] * videoElement.videoHeight,
          prediction.bbox[2] * videoElement.videoWidth,
          prediction.bbox[3] * videoElement.videoHeight,
        ],
      }));

      setDetectedClothesAboveThreshold(scaledPredictions);

      if (typeof imageTensor !== 'undefined') {
        tf.dispose(imageTensor);
      }
    } catch (error) {
      console.error("Error during clothing prediction:", error);
    }
  }, [modelRef, videoRef, isVideoReady, isTFReady, classNames]);

  const handleVideoPlay = useCallback(() => {
    const video = videoRef.current.video || videoRef.current;
    if (video) {
      setVideoSize({ width: video.videoWidth, height: video.videoHeight });
      setIsVideoReady(true);
    }
  }, []);

  useEffect(() => {
    let faceDetectionInterval;
    let clothingPredictionInterval;

    if (isModelsLoaded && videoRef.current && isVideoReady && isTFReady) {
      // Run face detection
      faceDetectionInterval = setInterval(async () => {
        const video = videoRef.current.video || videoRef.current;
        if (video && video.readyState === 4 && video.videoWidth > 0 && video.videoHeight > 0) {
          const displaySize = { width: video.videoWidth, height: video.videoHeight };
          const results = await faceapi
            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withAgeAndGender();
          const resizedResults = faceapi.resizeResults(results, displaySize);
          setDetections(resizedResults);
        }
      }, 100);

      // Run clothing prediction
      clothingPredictionInterval = setInterval(runClothingPrediction, 200);
    }

    return () => {
      clearInterval(faceDetectionInterval);
      clearInterval(clothingPredictionInterval);
    };
  }, [isModelsLoaded, videoRef, isVideoReady, runClothingPrediction, isTFReady]);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Face and Clothing Detection</h1>
      <div
        ref={containerRef}
        style={{
          position: "relative",
          display: "inline-block",
          width: `${videoSize.width}px`,
          height: `${videoSize.height}px`,
        }}
      >
        <Webcam
          ref={videoRef}
          width="100%"
          height="auto"
          onPlay={handleVideoPlay}
          autoPlay
          muted
          style={{ display: "block" }}
        />
        <AnimatePresence>
          {detections.map((det, i) => {
            const { x, y, width, height } = det.detection.box;
            const age = Math.round(det.age);
            const gender = det.gender;
            const genderProb = (det.genderProbability * 100).toFixed(0);

            return (
              <motion.div
                key={`face-${i}`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{
                  opacity: 1,
                  scale: 1,
                  left: x,
                  top: y,
                  width,
                  height,
                }}
                exit={{ opacity: 0 }}
                transition={{
                  duration: 0.1,
                  ease: "easeInOut",
                }}
                style={{
                  position: "absolute",
                  border: "2px solid #00ff88",
                  borderRadius: "8px",
                  zIndex: 2,
                  color: "#00ff88",
                  fontWeight: "bold",
                  fontSize: "12px",
                  padding: "2px",
                  backgroundColor: "rgba(0, 0, 0, 0.3)",
                  pointerEvents: "none",
                }}
              >
                {age} yrs | {gender} ({genderProb}%)
              </motion.div>
            );
          })}
        </AnimatePresence>
        {detectedClothesAboveThreshold.map((prediction, i) => {
          const [x, y, width, height] = prediction.scaledBBox;
          return (
            <div
              key={`clothing-${i}`}
              style={{
                position: "absolute",
                border: "2px solid #ff0000",
                borderRadius: "8px",
                zIndex: 2,
                color: "#ff0000",
                fontWeight: "bold",
                fontSize: "16px",
                padding: "8px",
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                pointerEvents: "none",
                textAlign: "center",
                left: x,
                top: y,
                width,
                height,
              }}
            >
              {prediction.className !== "NoDetect" ? `${prediction.className}: ${(prediction.probability * 100).toFixed(2)}%` : prediction.className}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default FaceDetection;
