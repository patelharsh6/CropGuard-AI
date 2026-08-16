'use client';

import { useState, useRef, useCallback } from 'react';
import { useCropGuardModel, InferenceResult } from '@/lib/useCropGuardModel';
import { CLASS_NAMES } from '@/lib/constants';
import { DISEASE_INFO } from '@/lib/diseaseInfo';

export default function Home() {
  const { isModelReady, status, error, infer } = useCropGuardModel();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [dropError, setDropError] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);

  // ─── Shared file entry point ───────────────────────────────────────────────
  // All three input paths (file input, camera capture, drag-and-drop) funnel
  // through this single function so preprocessing is never duplicated.
  const processFile = useCallback((file: File) => {
    setDropError(null);
    setResult(null);
    setInferenceError(null);

    if (!file.type.startsWith('image/')) {
      setDropError(`"${file.name}" is not an image file. Please drop a JPG, PNG, or WebP image.`);
      return;
    }

    const url = URL.createObjectURL(file);
    setSelectedImage(url);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
    // Reset the input value so the same file can be re-selected if needed
    e.target.value = '';
  };

  // ─── Drag-and-drop handlers ────────────────────────────────────────────────
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (isModelReady) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (!isModelReady) return;

    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    processFile(file);
  };

  // ─── Inference (triggered when the preview image finishes loading) ─────────
  const runInference = async () => {
    if (!imgRef.current || !isModelReady) return;

    setIsInferring(true);
    setInferenceError(null);
    setResult(null);

    try {
      const res = await infer(imgRef.current);
      setResult(res);
    } catch (err: unknown) {
      console.error('Inference error:', err);
      setInferenceError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsInferring(false);
    }
  };

  return (
    <main className="max-w-3xl mx-auto my-8 px-4 font-mono text-[#c8e6c9] bg-[#0f1b0e] min-h-screen">
      <h1 className="text-3xl font-bold text-[#66bb6a] mb-1">🌿 CropGuard AI</h1>
      <p className="text-gray-400 text-sm mb-6">
        Next.js browser inference — LiteRT.js · dynamic-range TFLite ·
        MobileNetV3Small · 17 classes
      </p>

      {/* Model status bar */}
      <div className={`p-3 mb-6 rounded border min-h-[3rem] ${error ? 'bg-[#1a0f0f] border-red-700 text-red-300' : isModelReady ? 'bg-[#111f11] border-[#2e7d32] text-[#a5d6a7]' : 'bg-[#1a2b1a] border-[#2d5a2d]'}`}>
        {error ? error : status}
      </div>

      {/* ── Drag-and-drop zone ────────────────────────────────────────────── */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={[
          'relative mb-6 rounded-lg border-2 border-dashed transition-colors duration-150 p-6',
          !isModelReady
            ? 'border-gray-700 bg-[#111] cursor-not-allowed opacity-50'
            : isDragOver
              ? 'border-[#66bb6a] bg-[#1a2e1a] cursor-copy'
              : 'border-[#2d5a2d] bg-[#111f11] hover:border-[#3d7a3d]',
        ].join(' ')}
      >
        <div className="text-center pointer-events-none select-none">
          <div className="text-4xl mb-2">{isDragOver ? '📥' : '🖼️'}</div>
          <p className="text-sm text-gray-400">
            {isDragOver ? 'Release to analyse' : 'Drag & drop a leaf image here'}
          </p>
          {!isDragOver && (
            <p className="text-xs text-gray-600 mt-1">JPG · PNG · WebP</p>
          )}
        </div>
      </div>

      {/* Drop validation error */}
      {dropError && (
        <div className="mb-4 bg-[#1a0f0f] border border-yellow-700 text-yellow-300 p-3 rounded text-sm">
          ⚠️ {dropError}
        </div>
      )}

      {/* ── Existing buttons (completely unchanged) ───────────────────────── */}
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

      {/* ── Image preview ─────────────────────────────────────────────────── */}
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

      {/* ── Prediction results ────────────────────────────────────────────── */}
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

      {/* ── Disease info panel ────────────────────────────────────────────── */}
      {result && DISEASE_INFO[result.bestClass] && (
        <div className="bg-[#111f11] border border-[#2d5a2d] rounded p-4 mt-4">
          <h2 className="text-[#66bb6a] font-bold text-xl mb-3">
            {DISEASE_INFO[result.bestClass].displayName}
          </h2>

          <div className="mb-4">
            <h3 className="text-[#81c784] font-semibold mb-1">Likely Cause:</h3>
            <p className="text-gray-300 text-sm">{DISEASE_INFO[result.bestClass].cause}</p>
          </div>

          <div className="mb-4">
            <h3 className="text-[#81c784] font-semibold mb-1">Prevention & Care:</h3>
            <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
              {DISEASE_INFO[result.bestClass].prevention.map((bullet, idx) => (
                <li key={idx}>{bullet}</li>
              ))}
            </ul>
          </div>

          <div className="mb-4">
            <h3 className="text-[#81c784] font-semibold mb-1">General Treatment Guidance:</h3>
            <p className="text-gray-300 text-sm">{DISEASE_INFO[result.bestClass].treatmentCategory}</p>
          </div>

          <div className="mt-6 pt-3 border-t border-[#1e3a1e] text-xs text-gray-400">
            <p>
              <span className="font-semibold text-gray-300">Disclaimer:</span> This is general information only and not a substitute for professional agricultural advice.
            </p>
            <p className="mt-1">
              For specific treatment guidance, see:{' '}
              <a
                href={DISEASE_INFO[result.bestClass].referenceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#66bb6a] hover:underline"
              >
                Extension Resource
              </a>
            </p>
          </div>
        </div>
      )}
    </main>
  );
}
