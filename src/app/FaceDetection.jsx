"use client";

import * as tf from "@tensorflow/tfjs";
import React, { useRef, useEffect, useState, useCallback } from "react";
import * as faceapi from "face-api.js";
import Webcam from "react-webcam";
import { motion, AnimatePresence } from "framer-motion";

// Assume your clothing model's output provides bounding box coordinates
// and class probabilities. The structure of this output is crucial.
// Example of expected output for each detected clothing item:
// {
//   bbox: [x, y, width, height], // Normalized coordinates (0 to 1)
//   className: "shirt",
//   probability: 0.85
// }

function FaceDetection() {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const modelRef = useRef(null); // TensorFlow model for clothing

  const [detections, setDetections] = useState([]);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [clothingPredictions, setClothingPredictions] = useState([]);
  const [classNames, setClassNames] = useState([]);
  const [isModelsLoaded, setIsModelsLoaded] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isTFReady, setIsTFReady] = useState(false);
  const [detectedClothesAboveThreshold, setDetectedClothesAboveThreshold] = useState([]);

  const predictionThreshold = 0.6; // 60% threshold

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = "/models/faceapi";
      const TF_MODEL_URL = "/models/clothing/model.json"; // Path to your TF clothing model
      const METADATA_URL = "/models/clothing/metadata.json"; // Path to your TF metadata

      try {
        // Explicitly set the TensorFlow.js backend
        // await tf.setBackend('webgl'); // Or 'cpu' if you prefer

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

      // Assuming your model's predict function returns an array of objects
      // where each object contains: { bbox: [x, y, width, height], score: ..., class: ... }
      const predictions = await modelRef.current.predict(imageTensor).data();

      // **Important:** You will need to process the raw `predictions` from your TensorFlow model here.
      // This processing will depend entirely on the output format of your model.
      // The goal is to extract bounding boxes, class names (using `classNames`), and confidence scores.

      // **Mock processing - Replace with your actual post-processing logic**
      const processedPredictions = [];
      // Example: Assuming your raw predictions are a flat array that needs reshaping
      // and association with class names and hypothetical bounding boxes.
      if (predictions && classNames.length > 0) {
        // This is a placeholder - adapt to your model's output structure
        for (let i = 0; i < classNames.length; i++) {
          const probability = predictions[i]; // Assuming a direct mapping for demonstration
          if (probability > predictionThreshold) {
            // Mock bounding box - your model should output these
            const x = Math.random() * (videoElement.videoWidth - 100);
            const y = Math.random() * (videoElement.videoHeight - 100);
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

      // Convert normalized bounding boxes to pixel coordinates
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
          {detectedClothesAboveThreshold.map((prediction, i) => {
            const [x, y, width, height] = prediction.scaledBBox;
            return (
              <motion.div
                key={`clothing-${i}`}
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
                }}
              >
                {prediction.className}: {(prediction.probability * 100).toFixed(2)}%
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default FaceDetection;