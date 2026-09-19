"""Competition entry point: read seal PNGs, write <team>.csv of seven-digit codes.

Runs from any working directory and always produces a CSV; unreadable images and
unexpected per-image errors fall back to a guess instead of aborting the run.
    python main.py --input-dir D --output-dir O --team NAME
"""
import sys
from pathlib import Path

# Make the bundled `seals` package importable however the script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import logging
from time import perf_counter

from seals.dataio import read_gray
from seals.recognize import FALLBACK_NUMBER, load_recognizer, predict_number

LOGGER = logging.getLogger(__name__)
# The trained model that ships with this folder.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'weights' / 'svm.joblib'
# About twenty progress updates over any folder size.
PROGRESS_UPDATES = 20


def find_images(directory):
    """Return the sorted .png files in a directory, case-insensitively."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == '.png')


def recognize_one(classifier, path):
    """Return (filename, number); never raises, so one bad image cannot abort the run."""
    try:
        gray = read_gray(path)
        if gray is None:
            return path.name, FALLBACK_NUMBER
        return path.name, predict_number(classifier, gray)
    except Exception:
        LOGGER.exception('Recognition failed for %s', path)
        return path.name, FALLBACK_NUMBER


def process_folder(classifier, image_paths):
    """Recognize every image one at a time and return ordered (filename, number) rows."""
    rows = []
    interval = max(1, len(image_paths) // PROGRESS_UPDATES)
    for index, path in enumerate(image_paths, start=1):
        rows.append(recognize_one(classifier, path))
        if index % interval == 0:
            LOGGER.info('Processed %d/%d images', index, len(image_paths))
    return rows


def write_csv(rows, output_path):
    """Write the `filename;number` CSV, header included, even when rows is empty."""
    with output_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(('filename', 'number'))
        writer.writerows(rows)


def parse_arguments():
    """Parse and resolve the command-line arguments."""
    parser = argparse.ArgumentParser(description='Classical seal-number OCR entry point.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--team', default='vision_crafters')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    arguments = parser.parse_args()
    arguments.input_dir = arguments.input_dir.resolve()
    arguments.output_dir = arguments.output_dir.resolve()
    return arguments


def main():
    """Recognize a folder of seals and write the submission CSV."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    arguments = parse_arguments()
    if not arguments.weights.is_file():
        LOGGER.error('Missing model weights: %s', arguments.weights)
        return 1
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / f'{arguments.team}.csv'
    image_paths = find_images(arguments.input_dir)
    if not image_paths:
        LOGGER.warning('No PNG images in %s; writing header-only CSV', arguments.input_dir)
        write_csv([], output_path)
        return 0
    classifier = load_recognizer(arguments.weights)
    rows, start = [], perf_counter()
    try:
        rows = process_folder(classifier, image_paths)
    finally:
        write_csv(rows, output_path)
    elapsed = perf_counter() - start
    fallbacks = sum(number == FALLBACK_NUMBER for _, number in rows)
    LOGGER.info('Wrote %d rows to %s in %.3f s (%.2f ms/image); %d fallback guesses',
                len(rows), output_path, elapsed, elapsed / len(rows) * 1000, fallbacks)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
