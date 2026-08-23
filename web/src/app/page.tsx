'use client';

import { useState, useRef, useCallback } from 'react';
import { useCropGuardModel, InferenceResult } from '@/lib/useCropGuardModel';
import { CLASS_NAMES } from '@/lib/constants';
import { DISEASE_INFO } from '@/lib/diseaseInfo';
import { getTier, getTopN, ConfidenceTier, TopPrediction } from '@/lib/confidenceTier';
import { TEMPERATURE } from '@/lib/calibration';

// ─── Sub-components ────────────────────────────────────────────────────────────

/** Expandable raw 17-class probability table. Always available regardless of tier. */
function RawOutputTable({ result }: { result: InferenceResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-4 pt-3 border-t border-[#1e3a1e]">
      <button
        onClick={() => setOpen(o => !o)}
        className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        {open ? '▲ Hide' : '▼ Show'} model output (17 classes)
      </button>
      {open && (
        <div className="mt-2 overflow-y-auto max-h-64 text-xs text-gray-400 whitespace-pre font-mono">
          {/* Column header aligned with the '[nn] ' + 35-char name + two 8-char
              numbers laid out below. */}
          <div className="text-gray-500">
            {''.padEnd(5 + 35)}{'calibrated'.padStart(8)} {'raw'.padStart(8)}  (T = {TEMPERATURE})
          </div>
          {CLASS_NAMES.map((name, i) => {
            const prob = result.probabilities[i];
            const raw = result.rawProbabilities[i];
            const bar = '█'.repeat(Math.round(prob * 30));
            return (
              <div key={i}>
                [{String(i).padStart(2, '0')}] {name.padEnd(35)} {prob.toFixed(6)} {raw.toFixed(6)}  {bar}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Compact ranked list of top-3 predictions with confidence bars. */
function Top3List({ predictions, labelAs }: { predictions: TopPrediction[]; labelAs: string }) {
  return (
    <div className="mt-3">
      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide">{labelAs}</p>
      <ol className="space-y-1">
        {predictions.map((p, rank) => (
          <li key={p.index} className="flex items-center gap-3 text-sm">
            <span className="text-gray-500 w-4 shrink-0">#{rank + 1}</span>
            <span className="text-gray-300 flex-1 truncate">{p.className.replace(/_/g, ' ')}</span>
            <span className={`font-mono text-xs ${rank === 0 ? 'text-[#a5d6a7]' : 'text-gray-500'}`}>
              {(p.confidence * 100).toFixed(1)}%
            </span>
            {/* Mini confidence bar */}
            <div className="w-20 h-1.5 bg-[#1e3a1e] rounded-full overflow-hidden shrink-0">
              <div
                className={`h-full rounded-full ${rank === 0 ? 'bg-[#66bb6a]' : 'bg-gray-600'}`}
                style={{ width: `${p.confidence * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Result panel — renders differently based on confidence tier. */
function ResultPanel({ result }: { result: InferenceResult }) {
  const tier: ConfidenceTier = getTier(result.bestConfidence);
  const top3 = getTopN(result.probabilities, 3);

  // ── HIGH tier ──────────────────────────────────────────────────────────────
  if (tier === 'HIGH') {
    return (
      <div className="mt-4 space-y-4">
        {/* Prediction card */}
        <div className="bg-[#111f11] border border-[#2d5a2d] rounded p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-bold px-2 py-0.5 rounded bg-[#1a3d1a] text-[#66bb6a] border border-[#2d7a3a] uppercase tracking-wide">
              ✓ High Confidence
            </span>
            <span className="text-xs text-gray-500">
              {(result.bestConfidence * 100).toFixed(2)}%
            </span>
          </div>
          <h2 className="text-xl font-bold text-[#a5d6a7] mb-1">
            {result.bestClass.replace(/_/g, ' ')}
          </h2>
          <div className="text-gray-500 text-xs">
            Preprocessing: {result.preprocMs} ms | Inference: {result.inferMs} ms
          </div>
          <RawOutputTable result={result} />
        </div>

        {/* Full treatment panel */}
        {DISEASE_INFO[result.bestClass] && (
          <div className="bg-[#111f11] border border-[#2d5a2d] rounded p-4">
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
                {DISEASE_INFO[result.bestClass].prevention.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
            <div className="mb-4">
              <h3 className="text-[#81c784] font-semibold mb-1">General Treatment Guidance:</h3>
              <p className="text-gray-300 text-sm">{DISEASE_INFO[result.bestClass].treatmentCategory}</p>
            </div>
            <div className="mt-6 pt-3 border-t border-[#1e3a1e] text-xs text-gray-400">
              <p>
                <span className="font-semibold text-gray-300">Disclaimer:</span>{' '}
                This is general information only and not a substitute for professional agricultural advice.
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
      </div>
    );
  }

  // ── MODERATE tier ──────────────────────────────────────────────────────────
  if (tier === 'MODERATE') {
    return (
      <div className="mt-4 space-y-4">
        {/* Prediction card with amber caution banner */}
        <div className="bg-[#111f11] border border-yellow-800 rounded p-4">
          {/* Caution banner */}
          <div className="flex items-start gap-2 mb-4 bg-yellow-900/30 border border-yellow-700/50 rounded px-3 py-2">
            <span className="text-yellow-400 mt-0.5 shrink-0">⚠</span>
            <p className="text-yellow-300 text-xs leading-relaxed">
              <span className="font-semibold">Moderate confidence</span> — verify against the
              alternatives below before treating. Retaking the photo with a single leaf in
              clear light may improve accuracy.
            </p>
          </div>

          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold px-2 py-0.5 rounded bg-yellow-900/40 text-yellow-400 border border-yellow-700 uppercase tracking-wide">
              ~ Moderate Confidence
            </span>
            <span className="text-xs text-gray-500">
              {(result.bestConfidence * 100).toFixed(2)}%
            </span>
          </div>
          <h2 className="text-xl font-bold text-[#a5d6a7] mb-1">
            {result.bestClass.replace(/_/g, ' ')}
          </h2>

          <Top3List predictions={top3} labelAs="Top alternatives" />
          <div className="text-gray-500 text-xs mt-3">
            Preprocessing: {result.preprocMs} ms | Inference: {result.inferMs} ms
          </div>
          <RawOutputTable result={result} />
        </div>

        {/* Treatment panel shown but with a reduced header */}
        {DISEASE_INFO[result.bestClass] && (
          <div className="bg-[#111f11] border border-yellow-800/50 rounded p-4">
            <p className="text-xs text-yellow-600 mb-3">
              General information for the most likely match — confirm diagnosis before applying treatments.
            </p>
            <h2 className="text-[#66bb6a] font-bold text-lg mb-3">
              {DISEASE_INFO[result.bestClass].displayName}
            </h2>
            <div className="mb-4">
              <h3 className="text-[#81c784] font-semibold mb-1">Likely Cause:</h3>
              <p className="text-gray-300 text-sm">{DISEASE_INFO[result.bestClass].cause}</p>
            </div>
            <div className="mb-4">
              <h3 className="text-[#81c784] font-semibold mb-1">Prevention & Care:</h3>
              <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                {DISEASE_INFO[result.bestClass].prevention.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
            <div className="mb-4">
              <h3 className="text-[#81c784] font-semibold mb-1">General Treatment Guidance:</h3>
              <p className="text-gray-300 text-sm">{DISEASE_INFO[result.bestClass].treatmentCategory}</p>
            </div>
            <div className="mt-6 pt-3 border-t border-[#1e3a1e] text-xs text-gray-400">
              <p>
                <span className="font-semibold text-gray-300">Disclaimer:</span>{' '}
                This is general information only and not a substitute for professional agricultural advice.
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
      </div>
    );
  }

  // ── LOW tier ───────────────────────────────────────────────────────────────
  return (
    <div className="mt-4 space-y-4">
      <div className="bg-[#111f11] border border-red-900 rounded p-4">
        {/* Uncertainty banner */}
        <div className="flex items-start gap-2 mb-4 bg-red-900/20 border border-red-800/50 rounded px-3 py-2">
          <span className="text-red-400 mt-0.5 shrink-0">✕</span>
          <p className="text-red-300 text-xs leading-relaxed">
            <span className="font-semibold">Uncertain result</span> — this photo doesn&apos;t clearly
            match a known disease pattern. Do not use this result for treatment decisions.
          </p>
        </div>

        <h2 className="text-lg font-bold text-gray-300 mb-1">
          Uncertain — no confident diagnosis
        </h2>
        <p className="text-gray-500 text-sm mb-4">
          Best guess: {result.bestClass.replace(/_/g, ' ')} ({(result.bestConfidence * 100).toFixed(1)}%)
        </p>

        <Top3List predictions={top3} labelAs="Possible matches (not a diagnosis)" />

        {/* Photo tips */}
        <div className="mt-4 bg-[#1a2b1a] border border-[#2d5a2d] rounded p-3">
          <p className="text-xs font-semibold text-[#81c784] mb-2">💡 Tips for a better result:</p>
          <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
            <li>Photograph a single leaf, not the whole plant</li>
            <li>Fill the frame with the affected leaf area</li>
            <li>Use bright, even natural light — avoid shadows and flash glare</li>
            <li>Hold the camera steady and focus on the lesion</li>
          </ul>
        </div>

        <p className="text-xs text-gray-500 mt-4">
          If the issue persists after retaking the photo, consult a{' '}
          <span className="text-[#66bb6a]">local agricultural extension agent</span> for
          an in-person diagnosis.
        </p>

        <div className="text-gray-600 text-xs mt-3">
          Preprocessing: {result.preprocMs} ms | Inference: {result.inferMs} ms
        </div>
        <RawOutputTable result={result} />
      </div>
      {/* NO treatment panel for LOW tier — treatment info implies a confident diagnosis */}
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const { isModelReady, status, error, infer } = useCropGuardModel();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [dropError, setDropError] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);

  // All three input paths funnel through processFile — preprocessing is never duplicated.
  const processFile = useCallback((file: File) => {
    setDropError(null);
    setResult(null);
    setInferenceError(null);

    if (!file.type.startsWith('image/')) {
      setDropError(`"${file.name}" is not an image file. Please use a JPG, PNG, or WebP image.`);
      return;
    }

    const url = URL.createObjectURL(file);
    setSelectedImage(url);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
    e.target.value = '';
  };

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
      <div className={`p-3 mb-6 rounded border min-h-[3rem] ${
        error
          ? 'bg-[#1a0f0f] border-red-700 text-red-300'
          : isModelReady
            ? 'bg-[#111f11] border-[#2e7d32] text-[#a5d6a7]'
            : 'bg-[#1a2b1a] border-[#2d5a2d]'
      }`}>
        {error ? error : status}
      </div>

      {/* Drag-and-drop zone */}
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
          {!isDragOver && <p className="text-xs text-gray-600 mt-1">JPG · PNG · WebP</p>}
        </div>
      </div>

      {dropError && (
        <div className="mb-4 bg-[#1a0f0f] border border-yellow-700 text-yellow-300 p-3 rounded text-sm">
          ⚠️ {dropError}
        </div>
      )}

      {/* Buttons */}
      <div className="mb-6 flex gap-4 flex-wrap">
        <label className={`inline-flex items-center px-5 py-2 rounded text-white cursor-pointer ${isModelReady ? 'bg-[#2d7a3a] hover:bg-[#388e3c]' : 'bg-gray-700 cursor-not-allowed'}`}>
          📸 Take Photo
          <input type="file" className="hidden" accept="image/*" capture="environment" disabled={!isModelReady} onChange={handleFileChange} />
        </label>
        <label className={`inline-flex items-center px-5 py-2 rounded text-white cursor-pointer ${isModelReady ? 'bg-[#1e3a1e] border border-[#2d7a3a] hover:bg-[#254625]' : 'bg-gray-700 cursor-not-allowed'}`}>
          📂 Choose from Gallery
          <input type="file" className="hidden" accept="image/*" disabled={!isModelReady} onChange={handleFileChange} />
        </label>
      </div>

      {/* Image preview */}
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

      {/* Tier-aware result panel */}
      {result && <ResultPanel result={result} />}
    </main>
  );
}
