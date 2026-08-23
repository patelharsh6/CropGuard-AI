import { CLASS_NAMES, ClassName } from './constants';

/** Confidence tiers based on top-1 softmax probability. */
export type ConfidenceTier = 'HIGH' | 'MODERATE' | 'LOW';

/**
 * DERIVED, not chosen. Produced by `python -m src.calibration` on the 3,625-image
 * *validation* split of temperature-scaled probabilities (see calibration.ts) and
 * then verified on the held-out test split. Report: outputs/calibration_report.json;
 * mirrored in src/config.py (TIER_HIGH / TIER_MODERATE).
 *
 * These thresholds apply to CALIBRATED probabilities — always run
 * `calibrateProbabilities()` first, or the tier is wrong.
 *
 * How each one was picked:
 *   HIGH     smallest tau whose selective accuracy holds >= 0.99 (and keeps
 *            holding for every larger tau). HIGH shows one diagnosis plus a
 *            treatment panel, so it is the tier that can do harm.
 *   MODERATE smallest tau where accuracy *locally* (within a 0.05-wide window,
 *            >= 50 images) stays >= 0.50 — below it, the top-1 is worse than a
 *            coin flip and the UI must stop naming a single disease.
 *
 * On the held-out test split (calibrated):
 *
 *   HIGH     (p >= 0.945)  74.4% coverage, 0.9930 top-1, 0.9993 top-3
 *   MODERATE (p >= 0.595)  20.9% coverage, 0.8393 top-1, 0.9895 top-3
 *   LOW                     4.7% coverage, 0.5529 top-1, 0.9588 top-3
 *
 * Confidently wrong (HIGH tier and incorrect) fell from 1.16% of all images
 * under the old intuited 0.85 / 0.50 pair to 0.52% — the safety metric the tier
 * system exists to hold down — at the cost of 8 points of HIGH coverage.
 * The LOW tier's 0.9588 top-3 is what justifies showing three candidates
 * there instead of nothing.
 *
 * Caveat that travels with these numbers: they are measured on the clean
 * PlantVillage distribution. On web-sourced field photos the gate degrades
 * (docs/DOMAIN_SHIFT.md), which calibration improves but does not fix.
 */
export const TIER_THRESHOLDS = {
  HIGH: 0.945,
  MODERATE: 0.595,
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
