'use client';

import { useState, useRef } from 'react';
import { useCropGuardModel, InferenceResult } from '@/lib/useCropGuardModel';
import { CLASS_NAMES } from '@/lib/constants';

export default function Home() {
  const { isModelReady, status, error, infer } = useCropGuardModel();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  
  const imgRef = useRef<HTMLImageElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset state
    setResult(null);
    setInferenceError(null);
    
    const url = URL.createObjectURL(file);
    setSelectedImage(url);
  };

  const runInference = async () => {
    if (!imgRef.current || !isModelReady) return;
    
    setIsInferring(true);
    setInferenceError(null);
    setResult(null);

    try {
      // Use the imported hook to run inference
      const res = await infer(imgRef.current);
      setResult(res);
    } catch (err: unknown) {
      console.error("Inference error:", err);
      setInferenceError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsInferring(false);
    }
  };

  return (
    <main className="max-w-3xl mx-auto my-8 px-4 font-mono text-[#c8e6c9] bg-[#0f1b0e] min-h-screen">
      <h1 className="text-3xl font-bold text-[#66bb6a] mb-1">🌿 CropGuard AI</h1>
      <p className="text-gray-400 text-sm mb-6">
        Next.js browser inference test — LiteRT.js · dynamic-range TFLite ·
        MobileNetV3Small · 17 classes
      </p>

      <div className={`p-3 mb-6 rounded border min-h-[3rem] ${error ? 'bg-[#1a0f0f] border-red-700 text-red-300' : isModelReady ? 'bg-[#111f11] border-[#2e7d32] text-[#a5d6a7]' : 'bg-[#1a2b1a] border-[#2d5a2d]'}`}>
        {error ? error : status}
      </div>

      <div className="mb-6 flex gap-4">
        <label className={`inline-flex items-center px-5 py-2 rounded text-white cursor-pointer ${isModelReady ? 'bg-[#2d7a3a] hover:bg-[#388e3c]' : 'bg-gray-700 cursor-not-allowed'}`}>
          📸 Take Photo
          <input 
            type="file" 
            className="hidden" 
            accept="image/*"
            capture="environment"
            disabled={!isModelReady}
            onChange={handleFileChange} 
          />
        </label>
        <label className={`inline-flex items-center px-5 py-2 rounded text-white cursor-pointer ${isModelReady ? 'bg-[#1e3a1e] border border-[#2d7a3a] hover:bg-[#254625]' : 'bg-gray-700 cursor-not-allowed'}`}>
          📂 Choose from Gallery
          <input 
            type="file" 
            className="hidden" 
            accept="image/*" 
            disabled={!isModelReady}
            onChange={handleFileChange} 
          />
        </label>
      </div>

      {selectedImage && (
        <div className="mb-6">
          <img 
            ref={imgRef}
            src={selectedImage} 
            alt="Selected leaf preview"
            className="w-56 h-56 object-cover border border-[#2d5a2d] mb-4"
            style={{ imageRendering: 'pixelated' }}
            onLoad={runInference}
          />
        </div>
      )}

      {isInferring && (
        <div className="text-[#a5d6a7] animate-pulse">Running inference...</div>
      )}

      {inferenceError && (
        <div className="bg-[#1a0f0f] border border-red-700 text-red-300 p-4 rounded">
          ❌ Inference failed: {inferenceError}
        </div>
      )}

      {result && (
        <div className="bg-[#111f11] border border-[#2d5a2d] rounded p-4 mt-4">
          <h2 className="text-[#81c784] font-bold text-lg mb-2">Prediction</h2>
          <div className="text-xl font-bold text-[#a5d6a7] mb-1">
            {result.bestClass} (class index {result.bestIndex})
          </div>
          <div className="text-gray-400 text-sm mb-3">
            Confidence: {(result.bestConfidence * 100).toFixed(2)}%
          </div>
          <div className="text-gray-500 text-xs mb-4">
            Preprocessing: {result.preprocMs} ms | Inference: {result.inferMs} ms
          </div>
          
          <div className="pt-2 border-t border-[#1e3a1e] overflow-y-auto max-h-64 text-xs text-gray-400 whitespace-pre">
            <div className="mb-2 font-bold">Raw softmax output (17 classes):</div>
            {CLASS_NAMES.map((name, i) => {
              const prob = result.probabilities[i];
              const bar = '█'.repeat(Math.round(prob * 30));
              return (
                <div key={i}>
                  [{String(i).padStart(2, '0')}] {name.padEnd(35)} {prob.toFixed(6)}  {bar}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </main>
  );
}
