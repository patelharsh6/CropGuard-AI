/**
 * Out-of-distribution gate — "is this even a leaf?"
 *
 * The classifier is closed-set over 17 leaf classes, so softmax always names a
 * disease: a chair, a face, or a whole tomato plant all get a diagnosis, and one
 * whole-plant photo scored 0.983 calibrated — above the HIGH gate. Calibration
 * (calibration.ts) cannot help, because it answers "how confident, *given a leaf*".
 * This runs before the diagnosis is shown and answers the prior question.
 *
 * Detector: cosine similarity between the model's 576-d penultimate embedding and
 * the nearest of the 17 class-mean embeddings, computed by `python -m src.ood` over
 * a 2,026-image sample of the training split. Score below `threshold` -> not a leaf.
 *
 * Why this and not the textbook detectors — measured in docs/OOD.md, AUROC against
 * the 101-image negative set:
 *
 *                    clean studio leaves   real field photos
 *   MSP                     0.9100              0.5949
 *   energy                  0.9275              0.6313
 *   Mahalanobis             0.9997              0.9649
 *   class-mean cosine       0.9997              0.9711   <- ships
 *
 * The right-hand column is the one that matters. Field photos are in-scope inputs a
 * gate must accept, and MSP/energy cannot tell them from a photo of a wall. Cosine
 * also costs 17x576 floats where Mahalanobis needs a 576x576 precision matrix —
 * 38 KB of parameters against 1.3 MB, on a 1.15 MB model. (As shipped JSON that is a
 * 96 KB fetch; the raw float32 figure is what the comparison is about.)
 *
 * The threshold likewise is NOT set on the clean validation split. That textbook
 * choice (95% TPR on studio images) rejects 95% of real field photos, because the
 * clean split is detached leaves on grey card and the gate learns the card. It is
 * set to keep 90% of the 37 field photos instead. n=37 is small: treat the exact
 * value as provisional and re-derive it when real_world_test/ grows.
 *
 * Params are fetched from /ood_gate.json, 96 KB (written by src/ood.py). Nothing here is
 * hand-tuned; edit the script and re-run rather than editing values.
 */

export interface OodGateParams {
  detector: string;
  embeddingDim: number;
  threshold: number;
  classNames: string[];
  /** Flattened [17 * 576] row-major, each row already L2-normalized. */
  unitMeans: Float32Array;
  fieldTpr: number;
  fieldN: number;
}

const GATE_URL = '/ood_gate.json';

// Module-level promise so React StrictMode's double-mount fetches once, matching
// how useCropGuardModel guards the WASM runtime init.
let gatePromise: Promise<OodGateParams> | null = null;

interface RawGateParams {
  detector: string;
  embedding_dim: number;
  threshold: number;
  class_names: string[];
  unit_means: number[][];
  field_tpr?: number;
  field_n?: number;
}

export function loadOodGate(url: string = GATE_URL): Promise<OodGateParams> {
  if (!gatePromise) {
    gatePromise = fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
        const raw = (await r.json()) as RawGateParams;
        const dim = raw.embedding_dim;
        const rows = raw.unit_means;
        if (!rows?.length || rows[0].length !== dim) {
          throw new Error('ood_gate.json: unit_means does not match embedding_dim');
        }
        // Flatten once: the per-inference cost is then 17 dot products over a
        // contiguous Float32Array rather than walking an array of arrays.
        const flat = new Float32Array(rows.length * dim);
        for (let c = 0; c < rows.length; c++) flat.set(rows[c], c * dim);
        return {
          detector: raw.detector,
          embeddingDim: dim,
          threshold: raw.threshold,
          classNames: raw.class_names,
          unitMeans: flat,
          fieldTpr: raw.field_tpr ?? 0.9,
          fieldN: raw.field_n ?? 0,
        };
      })
      .catch((err) => {
        // Let the next attempt retry rather than caching a rejected promise.
        gatePromise = null;
        throw err;
      });
  }
  return gatePromise;
}

export interface OodVerdict {
  /** max_c cos(embedding, classMean_c) — higher means more leaf-like. */
  score: number;
  /** Index of the nearest class mean. Diagnostic only; not the prediction. */
  nearestClass: number;
  threshold: number;
  /** True when the score is below threshold: do not show a diagnosis. */
  isOod: boolean;
}

/**
 * Score one embedding against the class means.
 *
 * Scale-invariance is the whole point: a field photo's embedding points nearly the
 * same direction as a studio leaf's but with a different magnitude, so cosine sees
 * a leaf where Mahalanobis sees an outlier.
 */
export function scoreEmbedding(
  embedding: Float32Array | number[],
  params: OodGateParams,
): OodVerdict {
  const dim = params.embeddingDim;
  if (embedding.length !== dim) {
    throw new Error(`embedding length ${embedding.length} != ${dim}`);
  }

  let norm = 0;
  for (let i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  // A zero embedding cannot happen after a ReLU-family backbone on a real image,
  // but a guard here is cheaper than a NaN reaching the UI.
  if (!(norm > 0)) {
    return { score: 0, nearestClass: 0, threshold: params.threshold, isOod: true };
  }

  let best = -Infinity;
  let bestClass = 0;
  const nClasses = params.unitMeans.length / dim;
  for (let c = 0; c < nClasses; c++) {
    const off = c * dim;
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += embedding[i] * params.unitMeans[off + i];
    if (dot > best) {
      best = dot;
      bestClass = c;
    }
  }

  const score = best / norm;
  return {
    score,
    nearestClass: bestClass,
    threshold: params.threshold,
    isOod: score < params.threshold,
  };
}
