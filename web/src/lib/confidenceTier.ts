import { CLASS_NAMES, ClassName } from './constants';

/** Confidence tiers based on top-1 softmax probability. */
export type ConfidenceTier = 'HIGH' | 'MODERATE' | 'LOW';

/**
 * Chosen by intuition, then measured with `python -m src.eval_tiers` over the
 * 3,625-image test split (outputs/tier_report.json):
 *
 *   HIGH     (p >= 0.85)  82.6% coverage, 0.9860 accuracy
 *   MODERATE (p >= 0.50)  14.9% coverage, 0.7681 accuracy
 *   LOW                    2.5% coverage, 0.4505 accuracy
 *
 * Validated, not yet calibrated — no ECE / temperature scaling, and the numbers
 * come from the test split of the same clean lab distribution the model trained
 * on. See plan.md Phase 2.
 */
export const TIER_THRESHOLDS = {
  HIGH: 0.85,
  MODERATE: 0.50,
} as const;

export interface TopPrediction {
  index: number;
  className: ClassName;
  confidence: number;
}

/** Classify a top-1 confidence value into a tier. */
export function getTier(topConfidence: number): ConfidenceTier {
  if (topConfidence >= TIER_THRESHOLDS.HIGH) return 'HIGH';
  if (topConfidence >= TIER_THRESHOLDS.MODERATE) return 'MODERATE';
  return 'LOW';
}

/** Return the top-N predictions sorted by confidence descending. */
export function getTopN(probabilities: number[], n = 3): TopPrediction[] {
  return probabilities
    .map((confidence, index) => ({
      index,
      className: CLASS_NAMES[index],
      confidence,
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}
