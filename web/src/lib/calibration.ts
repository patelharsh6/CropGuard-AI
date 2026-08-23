/**
 * Temperature scaling — the calibration step that turns the model's softmax
 * scores into numbers the UI is allowed to describe as probabilities.
 *
 * Measured by `python -m src.calibration` on the 3,625-image validation split,
 * using the same `cropguard_v1_production.tflite` this page loads. Full report:
 * outputs/calibration_report.json. Mirrored in src/config.py (TEMPERATURE).
 *
 *   validation   ECE 0.0152 -> 0.0084      (T fit here, by NLL minimization)
 *   test         ECE 0.0187 -> 0.0075      (held out — the honest number)
 *
 * T = 0.8878 is BELOW 1, i.e. the model was mildly *under*-confident: mean
 * confidence 0.9223 against 0.9401 accuracy on test. Sharpening is the opposite
 * of the textbook result for modern CNNs, and it is worth knowing why — the
 * two-phase fine-tune ran at a very low LR (1e-5 / 5e-6) with heavy realism
 * augmentation, which regularizes the logit scale rather than inflating it.
 *
 * Implementation: the production graph already ends in softmax, so logits are
 * not exposed. Since softmax(log(p)/T) === softmax((z - c)/T) for any constant
 * c, raising p to the power 1/T and renormalizing is exactly temperature
 * scaling on the hidden logits — no model change, no re-export, one pass over
 * 17 numbers.
 *
 * Note it is a strictly monotone transform, so the argmax NEVER changes:
 * calibration moves the confidence value and therefore the tier, never the
 * diagnosis itself.
 */
export const TEMPERATURE = 0.8878;

/**
 * Apply temperature scaling to a softmax probability vector.
 * Returns a new array; the input is untouched.
 */
export function calibrateProbabilities(
  probabilities: number[],
  temperature: number = TEMPERATURE,
): number[] {
  const invT = 1 / temperature;
  const scaled = probabilities.map((p) => Math.pow(Math.max(p, 0), invT));
  const total = scaled.reduce((a, b) => a + b, 0);
  // Degenerate only if every probability underflowed to 0, which the softmax
  // output cannot produce — fall back to the input rather than emitting NaNs.
  if (!(total > 0) || !Number.isFinite(total)) return [...probabilities];
  return scaled.map((p) => p / total);
}
