"""Harvest labeled digit crops from the seal train/val split.

Each seal is localized and its seven left-to-right boxes are mapped onto the seven
CSV digits, giving real in-domain training crops. Labels inherit any localization
error, so the harvest is noisy training data by design.
    python harvest.py --input-dir DIR --labels CSV --output-dir OUT
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2

from seals.config import DIGIT_COUNT
from seals.crops import crop_digit, normalize_digit
from seals.dataio import read_gray, read_manifest
from seals.localize import find_digits

LOGGER = logging.getLogger(__name__)
# One worker keeps saving deterministic; scans can opt into concurrency.
DEFAULT_WORKERS = 4
# Report progress about twenty times over any harvest size.
PROGRESS_UPDATES = 20


def harvest_image(source, image_directory):
    """Localize one seal and save its seven normalized crops; return (path, label) pairs."""
    filename, number = source
    gray = read_gray(filename)
    if gray is None:
        return []
    localization = find_digits(gray)
    if localization is None or len(localization.digit_boxes) != DIGIT_COUNT:
        return []
    crops = []
    for position, (box, digit) in enumerate(zip(localization.digit_boxes, number)):
        patch = crop_digit(gray, box, localization.angle)
        if patch.size == 0:
            return []
        name = f'{filename.stem}_{position}.png'
        cv2.imwrite(str(image_directory / name), normalize_digit(patch))
        crops.append((f'images/{name}', digit))
    return crops


def harvest_split(records, image_directory, workers):
    """Harvest every seal in parallel and collect all (path, label) crop pairs."""
    harvested = []
    interval = max(1, len(records) // PROGRESS_UPDATES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, batch in enumerate(executor.map(lambda r: harvest_image(r, image_directory), records), start=1):
            harvested.extend(batch)
            if index % interval == 0:
                LOGGER.info('Harvested %d/%d seals; %d crops so far', index, len(records), len(harvested))
    return harvested


def parse_arguments():
    """Parse and resolve the command-line arguments."""
    parser = argparse.ArgumentParser(description='Harvest labeled digit crops from localized seals.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 harvests every manifest record')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    arguments = parser.parse_args()
    if arguments.limit < 0 or arguments.workers < 1:
        parser.error('--limit must be nonnegative and --workers positive')
    for name in ('input_dir', 'labels', 'output_dir'):
        setattr(arguments, name, getattr(arguments, name).resolve())
    return arguments


def main():
    """Harvest crops from a split and write the harvest.csv manifest."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    arguments = parse_arguments()
    cv2.setNumThreads(1)
    records = read_manifest(arguments.labels, arguments.input_dir)
    if not records:
        LOGGER.error('No valid manifest records; harvest stopped')
        return 1
    if arguments.limit:
        records = records[:arguments.limit]
    image_directory = arguments.output_dir / 'images'
    image_directory.mkdir(parents=True, exist_ok=True)
    crops = harvest_split(records, image_directory, arguments.workers)
    manifest = arguments.output_dir / 'harvest.csv'
    with manifest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(('path', 'label'))
        writer.writerows(crops)
    LOGGER.info('Kept %d crops from %d seals -> %s', len(crops), len(records), manifest)
    return 0 if crops else 1


if __name__ == '__main__':
    raise SystemExit(main())
