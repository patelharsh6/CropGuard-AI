import { INPUT_H, INPUT_W } from './constants';

/**
 * Preprocess a loaded HTMLImageElement to a flat Float32Array [1,224,224,3].
 *
 * Resize strategy: drawImage onto an offscreen 224×224 Canvas.
 * Canvas uses bilinear interpolation by default.
 * WARNING: This logic has been verified against Python and must not be altered!
 * Explicitly setting imageSmoothingQuality to 'medium' matches TensorFlow's
 * tf.image.resize(method='bilinear'), which is used in the Python pipeline.
 *
 * Normalization: pixel / 255.0 -> [0, 1] (no channel mean subtraction)
 */
export function preprocessImage(imgEl: HTMLImageElement): Float32Array {
  // Use a standard canvas if OffscreenCanvas is unavailable (safari edge cases),
  // but OffscreenCanvas is preferred for performance without DOM painting.
  let canvas: HTMLCanvasElement | OffscreenCanvas;
  let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;

  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(INPUT_W, INPUT_H);
    ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  } else {
    canvas = document.createElement('canvas');
    canvas.width = INPUT_W;
    canvas.height = INPUT_H;
    ctx = canvas.getContext('2d');
  }

  if (!ctx) {
    throw new Error('Could not get 2D context for image preprocessing.');
  }

  // Explicitly set imageSmoothingQuality to 'medium' (bilinear) to match TF.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';

  ctx.drawImage(imgEl, 0, 0, INPUT_W, INPUT_H);
  
  const { data } = ctx.getImageData(0, 0, INPUT_W, INPUT_H);

  // Flatten to float32 NHWC [1, H, W, 3], dropping the alpha channel
  const float32 = new Float32Array(1 * INPUT_H * INPUT_W * 3);
  let fi = 0;
  for (let i = 0; i < data.length; i += 4) {
    float32[fi++] = data[i] / 255.0;     // R
    float32[fi++] = data[i + 1] / 255.0; // G
    float32[fi++] = data[i + 2] / 255.0; // B
  }
  return float32;
}
