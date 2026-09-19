"""Score the pipeline end to end against labeled seals.

Reports exact-match rate, per-digit accuracy, and failure attribution (localizer
found no row vs. recognizer misread), measuring the exact path main.py ships.
    python eval.py --input-dir DIR --labels CSV
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
from time import perf_counter

import cv2
import numpy as np

from seals.config import DIGIT_COUNT
from seals.dataio import read_gray, read_manifest
from seals.recognize import FALLBACK_NUMBER, load_recognizer, predict_with_status

LOGGER = logging.getLogger(__name__)
# The shipped model path.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'weights' / 'svm.joblib'


def evaluate(classifier, records):
    """Recognize every labeled seal and aggregate accuracy, errors, and timing."""
    matched = recognizer_errors = localizer_misses = correct_digits = 0
    timings = []
    for filename, expected in records:
        gray = read_gray(filename)
        if gray is None:
            predicted, status = FALLBACK_NUMBER, 'fallback'
        else:
            start = perf_counter()
            predicted, status = predict_with_status(classifier, gray)
            timings.append((perf_counter() - start) * 1000)
        correct_digits += sum(guess == truth for guess, truth in zip(predicted, expected))
        if predicted == expected:
            matched += 1
        elif status == 'fallback':
            localizer_misses += 1
        else:
            recognizer_errors += 1
    total = len(records)
    return dict(seals=total, exact_match=matched, exact_match_rate=round(matched / total, 4),
                digit_accuracy=round(correct_digits / (total * DIGIT_COUNT), 4),
                localizer_misses=localizer_misses, recognizer_errors=recognizer_errors,
                localization_p50_ms=round(float(np.median(timings)), 3) if timings else None,
                localization_p95_ms=round(float(np.percentile(timings, 95)), 3) if timings else None)


def parse_arguments():
    """Parse and resolve the command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate classical seal OCR against labels.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--report', type=Path)
    arguments = parser.parse_args()
    for name in ('input_dir', 'labels', 'report'):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.resolve())
    return arguments


def main():
    """Evaluate the model on a labeled split and optionally save a JSON report."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    arguments = parse_arguments()
    cv2.setNumThreads(1)
    records = read_manifest(arguments.labels, arguments.input_dir)
    if arguments.limit:
        records = records[:arguments.limit]
    if not records or not arguments.weights.is_file():
        LOGGER.error('Need a valid manifest and model weights')
        return 1
    summary = evaluate(load_recognizer(arguments.weights), records)
    LOGGER.info('Exact match %d/%d (%.4f); digit accuracy %.4f; localizer misses %d; p50 %.3f ms',
                summary['exact_match'], summary['seals'], summary['exact_match_rate'],
                summary['digit_accuracy'], summary['localizer_misses'], summary['localization_p50_ms'] or 0)
    if arguments.report:
        arguments.report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
