import { useState, useEffect, useRef } from 'react';
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';
import { INPUT_H, INPUT_W, CLASS_NAMES, ClassName } from './constants';
import { preprocessImage } from './preprocess';
import { calibrateProbabilities } from './calibration';
import { loadOodGate, scoreEmbedding, OodGateParams, OodVerdict } from './oodGate';

const WASM_PATH = '/litert_wasm/';
// Three outputs: softmax probabilities, raw logits, and the 576-d penultimate
// embedding the OOD gate scores (oodGate.ts). Same weights and the same
// dynamic-range quantization as cropguard_v1_production.tflite — verified
// bit-identical on probabilities by `python -m src.export_ood_model`
// (outputs/gate_export_report.json), so the diagnosis is unchanged by the swap.
const MODEL_URL = '/cropguard_v1_gate.tflite';

// Module-level promise to ensure we only load the WASM runtime once,
// preventing issues with React StrictMode double-mounting.
let liteRtInitPromise: Promise<unknown> | null = null;

/** Which model output is which. TFLite orders outputs by internal tensor index,
 *  not by the order the Keras model declared them, and the exported names are
 *  opaque — so they are resolved by shape, with the two 17-wide outputs told apart
 *  by which one sums to 1. */
interface OutputSlots {
  probs: number;
  logits: number;
  embedding: number;
}

function resolveSlots(outputs: Float32Array[]): OutputSlots {
  let probs = -1;
  let logits = -1;
  let embedding = -1;
  outputs.forEach((arr, i) => {
    if (arr.length > CLASS_NAMES.length) {
      embedding = i;
      return;
    }
    const sum = arr.reduce((a, b) => a + b, 0);
    const nonNegative = arr.every((v) => v >= 0);
    if (nonNegative && Math.abs(sum - 1) < 1e-3) probs = i;
    else logits = i;
  });
  if (probs < 0 || embedding < 0) {
    throw new Error('Model outputs not recognised — expected probs + embedding');
  }
  return { probs, logits, embedding };
}

export interface InferenceResult {
  bestClass: ClassName;
  bestIndex: number;
  /** Calibrated top-1 probability — this is what the tier thresholds expect. */
  bestConfidence: number;
  preprocMs: string;
  inferMs: string;
  /** Temperature-scaled probabilities (see calibration.ts). */
  probabilities: number[];
  /** Raw model softmax, before temperature scaling — kept for the debug panel. */
  rawProbabilities: number[];
  /**
   * OOD verdict, or null if /ood_gate.json could not be loaded. Null means
   * "unknown", not "in distribution" — the UI degrades to the tier system alone
   * rather than silently claiming every input is a leaf.
   */
  ood: OodVerdict | null;
}

export function useCropGuardModel() {
  const [isModelReady, setIsModelReady] = useState(false);
  const [status, setStatus] = useState<string>('Initializing LiteRT runtime...');
  const [error, setError] = useState<string | null>(null);
  /** Non-fatal: the classifier still works, only the gate is missing. */
  const [gateError, setGateError] = useState<string | null>(null);

  // We use a ref to hold the model instance to avoid React state re-renders with the non-serializable object
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);
  const gateRef = useRef<OodGateParams | null>(null);
  const slotsRef = useRef<OutputSlots | null>(null);

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
        setStatus('Loading & compiling model (1.14 MB)...');

        // Compile using the XNNPACK (wasm) accelerator to keep activations float32
        const compiledModel = await loadAndCompile(MODEL_URL, { accelerator: 'wasm' });

        if (!isMounted) return;
        modelRef.current = compiledModel;

        // The gate is a separate 96 KB fetch and must not block a working
        // classifier: on failure the UI keeps running without OOD rejection.
        setStatus('Loading OOD gate parameters...');
        try {
          const gate = await loadOodGate();
          if (isMounted) gateRef.current = gate;
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          console.warn('OOD gate unavailable:', err);
          if (isMounted) setGateError(msg);
        }

        if (!isMounted) return;
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

    const results = await modelRef.current.run(inputTensor);
    const inferMs = (performance.now() - t1).toFixed(1);

    // 3. Read outputs
    // For 'wasm' accelerator, output is already on the wasm backend — no moveTo needed.
    const arrays = (results as { toTypedArray: () => Float32Array }[]).map(
      (t) => t.toTypedArray() as Float32Array,
    );
    if (!slotsRef.current) slotsRef.current = resolveSlots(arrays);
    const slots = slotsRef.current;

    const rawProbabilities = Array.from(arrays[slots.probs]);

    // 4. OOD gate, on the embedding — before anything reads a confidence. The
    //    diagnosis is still computed either way; the UI decides what to show.
    let ood: OodVerdict | null = null;
    if (gateRef.current) {
      ood = scoreEmbedding(arrays[slots.embedding], gateRef.current);
    }

    // Temperature scaling, measured on the validation split. Monotone, so the
    // argmax below is unaffected — only the confidence value and hence the tier.
    const probabilities = calibrateProbabilities(rawProbabilities);

    // 5. Cleanup tensors to avoid Wasm heap leaks
    inputTensor.delete();
    (results as { delete: () => void }[]).forEach((t) => t.delete());

    // 6. Calculate argmax
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
      rawProbabilities,
      ood,
    };
  };

  return { isModelReady, status, error, gateError, infer };
}
