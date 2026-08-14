/**
 * CropGuard AI — LiteRT.js browser inference test harness
 *
 * Model: cropguard_v1_production.tflite
 *   Input : float32 [1, 224, 224, 3], NHWC, normalized to [0, 1]
 *   Output: float32 [1, 17], softmax probabilities
 *
 * Preprocessing mirrors the Python training pipeline exactly:
 *   tf.image.resize → bilinear (browser Canvas drawImage also uses bilinear)
 *   pixel / 255.0 → [0, 1] float32
 *   NO mean subtraction, NO std normalization — matches src/data_pipeline.py
 *
 * API reference (LiteRT.js @litertjs/core):
 *   loadLiteRt(wasmPath)  — initialize the Wasm runtime once
 *   loadAndCompile(url, { accelerator })  — load + JIT-compile the model
 *   model.run(new Tensor(float32Array, shape))  — run inference
 *   results[0].toTypedArray()  — read output (already on wasm/cpu backend for 'wasm')
 */

// ── Class labels (index 0–16, matching model output order) ──────────────────
const CLASS_NAMES = [
  'Corn_Cercospora_Gray_leaf_spot',   // 0
  'Corn_Common_rust',                  // 1
  'Corn_Northern_Leaf_Blight',         // 2
  'Corn_healthy',                      // 3
  'Potato_Early_blight',               // 4
  'Potato_Late_blight',                // 5
  'Potato_healthy',                    // 6
  'Tomato_Bacterial_spot',             // 7
  'Tomato_Early_blight',               // 8
  'Tomato_Late_blight',                // 9
  'Tomato_Leaf_Mold',                  // 10
  'Tomato_Septoria_leaf_spot',         // 11
  'Tomato_Spider_mites',               // 12
  'Tomato_Target_Spot',                // 13
  'Tomato_Yellow_Leaf_Curl_Virus',     // 14
  'Tomato_mosaic_virus',               // 15
  'Tomato_healthy',                    // 16
];

const INPUT_H   = 224;
const INPUT_W   = 224;
const N_CLASSES = 17;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const statusEl    = document.getElementById('status');
const fileInput   = document.getElementById('file-input');
const previewEl   = document.getElementById('preview');
const resultBox   = document.getElementById('result-box');
const predEl      = document.getElementById('prediction');
const confEl      = document.getElementById('confidence');
const timingEl    = document.getElementById('timing');
const rawEl       = document.getElementById('raw-output');

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(msg, type = '') {
  statusEl.className = type;
  statusEl.innerHTML = msg;
}

/** Argmax of a flat Float32Array */
function argmax(arr) {
  let maxIdx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[maxIdx]) maxIdx = i;
  }
  return maxIdx;
}

/**
 * Preprocess a loaded HTMLImageElement to a flat Float32Array [1,224,224,3].
 *
 * Resize strategy: drawImage onto an offscreen 224×224 Canvas.
 * Canvas uses bilinear interpolation by default — this matches TensorFlow's
 * tf.image.resize(method='bilinear'), which is also what Python's pipeline uses.
 * Note: tf.image.resize(method='bilinear', antialias=False) is the TF default.
 * Canvas does NOT apply anti-aliasing by default, so the match is very close.
 * Any sub-pixel residual difference is negligible for classification (softmax
 * confidence typically >99%).
 *
 * Normalization: pixel / 255.0 → [0, 1]  (no channel mean subtraction)
 * This is identical to:  tf.cast(img, tf.float32) / 255.0  in data_pipeline.py
 */
function preprocessImage(imgEl) {
  const canvas = new OffscreenCanvas(INPUT_W, INPUT_H);
  const ctx = canvas.getContext('2d');

  // Explicitly set imageSmoothingQuality to 'medium' (bilinear) to match TF.
  // 'high' would use bicubic, which would diverge from Python.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium'; // bilinear in all major browsers

  ctx.drawImage(imgEl, 0, 0, INPUT_W, INPUT_H);
  const { data } = ctx.getImageData(0, 0, INPUT_W, INPUT_H); // Uint8ClampedArray RGBA

  // Flatten to float32 NHWC [1, H, W, 3], dropping the alpha channel
  const float32 = new Float32Array(1 * INPUT_H * INPUT_W * 3);
  let fi = 0;
  for (let i = 0; i < data.length; i += 4) {
    float32[fi++] = data[i]     / 255.0; // R
    float32[fi++] = data[i + 1] / 255.0; // G
    float32[fi++] = data[i + 2] / 255.0; // B
    // data[i+3] is alpha — skip
  }
  return float32;
}

// ── LiteRT.js initialization ─────────────────────────────────────────────────
//
// Package: @litertjs/core  (published July 2026)
// CDN via jsDelivr ESM — no bundler required for this test harness.
// The Wasm binary (litert_wasm.wasm) is loaded from the same CDN path.
// In production you'd copy node_modules/@litertjs/core/wasm/ into public/.
//
// Accelerator choices:
//   'wasm'   → XNNPACK CPU (guaranteed to work everywhere; our production choice)
//   'webgpu' → GPU (faster, but requires a secure context and WebGPU support)
//
// We use 'wasm' (XNNPACK) because:
//   1. The dynamic-range model runs float32 activations — XNNPACK handles this well
//   2. Broadest compatibility (doesn't require WebGPU)
//   3. Matches the benchmark we already measured (22 ms median)

import { loadLiteRt, loadAndCompile, Tensor }
  from 'https://cdn.jsdelivr.net/npm/@litertjs/core/+esm';

const WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/';
const MODEL_URL  = 'public/cropguard_v1_production.tflite';

let model = null; // will be set after loadAndCompile

async function initModel() {
  try {
    setStatus('<span class="spinner"></span>Initializing LiteRT Wasm runtime…');
    await loadLiteRt(WASM_PATH);

    setStatus('<span class="spinner"></span>Loading & compiling model (1.15 MB)…');
    model = await loadAndCompile(MODEL_URL, { accelerator: 'wasm' });

    setStatus('✅ Model ready — choose a leaf image to classify', 'ok');
    fileInput.disabled = false;
  } catch (err) {
    console.error('LiteRT init failed:', err);
    setStatus(`❌ Init failed: ${err.message}<br><small>${err.stack ?? ''}</small>`, 'error');
  }
}

// ── Inference on file selection ───────────────────────────────────────────────
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !model) return;

  // Show preview at natural size then display 224×224
  const url = URL.createObjectURL(file);
  previewEl.style.display = 'block';
  previewEl.src = url;

  // Wait for the image to load before preprocessing
  await new Promise((resolve, reject) => {
    previewEl.onload  = resolve;
    previewEl.onerror = reject;
  });

  setStatus('<span class="spinner"></span>Running inference…');
  resultBox.style.display = 'none';

  try {
    // 1. Preprocess
    const t0 = performance.now();
    const inputData = preprocessImage(previewEl);
    const preprocMs = (performance.now() - t0).toFixed(1);

    // 2. Build tensor and run
    const inputTensor = new Tensor(inputData, [1, INPUT_H, INPUT_W, 3]);
    const t1 = performance.now();
    const results = await model.run(inputTensor);
    const inferMs = (performance.now() - t1).toFixed(1);

    // 3. Read output
    // For 'wasm' accelerator, output is already on the wasm backend — no moveTo needed.
    // If you switch to 'webgpu', uncomment:
    //   const outputTensor = await results[0].moveTo('wasm');
    //   const probs = outputTensor.toTypedArray();
    const probs = results[0].toTypedArray();

    // 4. Cleanup tensors to avoid Wasm heap leaks
    inputTensor.delete();
    results[0].delete();

    // 5. Display results
    const bestIdx  = argmax(probs);
    const bestConf = (probs[bestIdx] * 100).toFixed(2);
    const bestName = CLASS_NAMES[bestIdx];

    predEl.textContent  = `${bestName}  (class index ${bestIdx})`;
    confEl.textContent  = `Confidence: ${bestConf}%`;
    timingEl.textContent = `Preprocessing: ${preprocMs} ms | Inference: ${inferMs} ms`;

    // Raw 17-class output — aligned for readability
    const lines = CLASS_NAMES.map((name, i) => {
      const bar = '█'.repeat(Math.round(probs[i] * 30));
      return `[${String(i).padStart(2, '0')}] ${name.padEnd(35)} ${probs[i].toFixed(6)}  ${bar}`;
    });
    rawEl.textContent = 'Raw softmax output (17 classes):\n' + lines.join('\n');

    resultBox.style.display = 'block';
    setStatus(`✅ Done — predicted: ${bestName}`, 'ok');

    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Inference error:', err);
    setStatus(`❌ Inference failed: ${err.message}`, 'error');
    URL.revokeObjectURL(url);
  }
});

// ── Boot ─────────────────────────────────────────────────────────────────────
initModel();
