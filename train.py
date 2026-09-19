"""Train the augmented RBF-SVM digit recognizer on harvested crops.

Training crops come from the seal train split and always include augmented copies;
validation crops come from the val split, so no seal contributes to both.
    python train.py --train harvest_train/harvest.csv --val harvest_val/harvest.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from seals.config import DEFAULT_SEED
from seals.dataset import balance, extract_features, read_pairs

LOGGER = logging.getLogger(__name__)
# The shipped model path.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'weights' / 'svm.joblib'
# Cap crops per class so RBF training stays tractable and less imbalanced.
DEFAULT_CAP = 2500
# Augmented copies added per crop; augmentation is always on for this model.
AUGMENT_COPIES = 1
# RBF penalty tuned for this single fixed font.
SVM_C = 10.0


def build_classifier():
    """Return the StandardScaler -> RBF-SVC pipeline used for recognition."""
    return make_pipeline(StandardScaler(),
                         SVC(kernel='rbf', C=SVM_C, gamma='scale', cache_size=2000))


def evaluate_crops(classifier, manifest):
    """Report per-crop accuracy and a confusion matrix on a clean crop manifest."""
    feature_matrix, labels = extract_features(read_pairs(manifest))
    predicted = classifier.predict(feature_matrix)
    accuracy = float((predicted == labels).mean())
    matrix = confusion_matrix(labels, predicted, labels=list(range(10)))
    return accuracy, matrix.tolist(), len(labels)


def parse_arguments():
    """Parse and resolve the command-line arguments."""
    parser = argparse.ArgumentParser(description='Train the augmented classical SVM digit recognizer.')
    parser.add_argument('--train', type=Path, required=True, help='harvest.csv from the train split')
    parser.add_argument('--val', type=Path, help='harvest.csv from the val split for reporting')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--cap', type=int, default=DEFAULT_CAP)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    for name in ('train', 'val', 'weights'):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.resolve())
    return arguments


def main():
    """Fit the augmented SVM, persist it, and write a small training report."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    import joblib
    arguments = parse_arguments()
    pairs = balance(read_pairs(arguments.train), arguments.cap, arguments.seed)
    if not pairs:
        LOGGER.error('No training crops found in %s', arguments.train)
        return 1
    LOGGER.info('Extracting features for %d base crops (+%d augmented copies each)', len(pairs), AUGMENT_COPIES)
    feature_matrix, labels = extract_features(pairs, copies=AUGMENT_COPIES, seed=arguments.seed)
    classifier = build_classifier()
    LOGGER.info('Fitting SVM on %s features', feature_matrix.shape)
    classifier.fit(feature_matrix, labels)
    arguments.weights.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, arguments.weights)
    report = {'base_crops': len(pairs), 'train_samples': int(len(labels)), 'augment_copies': AUGMENT_COPIES,
              'features': int(feature_matrix.shape[1]), 'cap': arguments.cap, 'seed': arguments.seed}
    if arguments.val:
        accuracy, matrix, count = evaluate_crops(classifier, arguments.val)
        report.update(val_crops=count, val_accuracy=round(accuracy, 4), val_confusion=matrix)
        LOGGER.info('Validation per-crop accuracy %.4f on %d crops', accuracy, count)
    arguments.weights.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    LOGGER.info('Saved model to %s', arguments.weights)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
