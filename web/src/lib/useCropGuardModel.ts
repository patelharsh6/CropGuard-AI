import { useState, useEffect, useRef } from 'react';
// @ts-expect-error Types are inferred during runtime loading for @litertjs/core
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';
import { INPUT_H, INPUT_W, CLASS_NAMES, ClassName } from './constants';
import { preprocessImage } from './preprocess';

const WASM_PATH = '/litert_wasm/';
const MODEL_URL = '/cropguard_v1_production.tflite';

// Module-level promise to ensure we only load the WASM runtime once,
// preventing issues with React StrictMode double-mounting.
let liteRtInitPromise: Promise<unknown> | null = null;

export interface InferenceResult {
  bestClass: ClassName;
  bestIndex: number;
  bestConfidence: number;
  preprocMs: string;
  inferMs: string;
  probabilities: number[];
}

export function useCropGuardModel() {
  const [isModelReady, setIsModelReady] = useState(false);
  const [status, setStatus] = useState<string>('Initializing LiteRT runtime...');
  const [error, setError] = useState<string | null>(null);

  // We use a ref to hold the model instance to avoid React state re-renders with the non-serializable object
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);

  useEffect(() => {
    let isMounted = true;

    async function initModel() {
      try {
        setStatus('Initializing LiteRT Wasm runtime...');
        if (!liteRtInitPromise) {
          liteRtInitPromise = loadLiteRt(WASM_PATH);
        }
        await liteRtInitPromise;

        if (!isMounted) return;
        setStatus('Loading & compiling model (1.15 MB)...');
        
        // Compile using the XNNPACK (wasm) accelerator to keep activations float32
        const compiledModel = await loadAndCompile(MODEL_URL, { accelerator: 'wasm' });
        
        if (!isMounted) return;
        modelRef.current = compiledModel;
        setIsModelReady(true);
        setStatus('Ready — choose a leaf image to classify');
      } catch (err: unknown) {
        if (!isMounted) return;
        const msg = err instanceof Error ? err.message : String(err);
        console.error('LiteRT init failed:', err);
        setError(`Init failed: ${msg}`);
        setStatus('Error during initialization');
      }
    }

    initModel();

    return () => {
      isMounted = false;
      // Note: we don't aggressively unloadLiteRt here to allow fast remounts in dev, 
      // but in a more complex app we might manage the WASM memory lifecycle more tightly.
    };
  }, []);

  const infer = async (imgEl: HTMLImageElement): Promise<InferenceResult> => {
    if (!modelRef.current) {
      throw new Error("Model not ready");
    }

    // 1. Preprocess
    const t0 = performance.now();
    const inputData = preprocessImage(imgEl);
    const preprocMs = (performance.now() - t0).toFixed(1);

    // 2. Build tensor and run
    const inputTensor = new Tensor(inputData, [1, INPUT_H, INPUT_W, 3]);
    const t1 = performance.now();
    
    // Note: this casts the model output correctly for our single output tensor
    const results = await modelRef.current.run(inputTensor);
    const inferMs = (performance.now() - t1).toFixed(1);

    // 3. Read output
    // For 'wasm' accelerator, output is already on the wasm backend — no moveTo needed.
    const probsArray = results[0].toTypedArray() as Float32Array;
    const probabilities = Array.from(probsArray);

    // 4. Cleanup tensors to avoid Wasm heap leaks
    inputTensor.delete();
    results[0].delete();

    // 5. Calculate argmax
    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) {
        bestIndex = i;
      }
    }

    const bestConfidence = probabilities[bestIndex];
    const bestClass = CLASS_NAMES[bestIndex];

    return {
      bestClass,
      bestIndex,
      bestConfidence,
      preprocMs,
      inferMs,
      probabilities,
    };
  };

  return { isModelReady, status, error, infer };
}
