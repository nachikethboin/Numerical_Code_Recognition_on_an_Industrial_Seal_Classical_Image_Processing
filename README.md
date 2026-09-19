# Numerical_Code_Recognition_on_an_Industrial_Seal_Using_Classical_Image_Processing

Reads the seven-digit code stamped on an industrial seal from a 1920×1200 grayscale
PNG and writes it to a CSV. Pure classical computer vision — localize the digit row,
cut and normalize each digit, and classify it with an RBF SVM on handcrafted features.
No GPU and no network required.

This folder is **self-contained**: the localization stack it once shared with the
wider project is copied into the `seals/` package, so nothing outside this folder is
imported.

## Run

```bash
python main.py --input-dir <folder-of-pngs> --output-dir <out> --team <name>
```

Writes `<out>/<name>.csv` with header `filename;number`. Runs from any working
directory, processes the images one at a time, and always writes a CSV — unreadable
images get a fallback guess rather than aborting the run.

## Install

```bash
pip install -r requirements.txt
```

The complete pinned set (numpy, opencv-python, scikit-learn, scipy, joblib,
threadpoolctl) — the model in `weights/svm.joblib` loads under exactly these versions.

## Layout

```
main.py        competition entry point (folder of PNGs -> <team>.csv)
harvest.py     build labeled digit crops from a localized split
train.py       train the augmented SVM from harvested crops
eval.py        score end to end against labels (exact match, per-digit, attribution)
requirements.txt
weights/       svm.joblib + svm.report.json (the shipped model)
seals/         the library package
  config.py      constants (digit count, angle limit, seed)
  dataio.py      read images and label manifests
  thresholds.py  polarity/method binary masks       (localization)
  rows.py        group components into digit rows    (localization)
  localize.py    find the seven digit boxes          (localization)
  crops.py       cut + normalize one digit (shared by training and inference)
  augment.py     label-preserving crop augmentation (training only)
  features.py    HOG + zoning + holes + aspect = 343-length vector
  recognize.py   localize -> classify -> seven-digit string
  dataset.py     load harvested crops into a feature matrix
```

## Pipeline

`PNG -> read_gray -> find_digits -> crop_digit ×7 -> normalize_digit ×7 -> features ×7
-> StandardScaler -> SVC(rbf) -> 7 digits -> CSV`

The same `normalize_digit` is applied to training crops and inference crops, which
removes the domain gap. Localization runs a staged threshold cascade (Otsu → adaptive,
then line-aligned recovery) and rejects the `TESCO` text and hardware by geometry.

## Model

`weights/svm.joblib` is an RBF SVM trained on crops **harvested** from the seal train
split (only 1,320 of the 48,620 listed `.tif` crops exist locally, so real crops are
harvested by localizing each seal and mapping its seven boxes onto the seven CSV
digits). Training uses 25,000 base crops (capped 2,500/class) plus one augmented copy
each (50,000 samples); augmentation is rotation ±8°, scale 0.9–1.1, small shift,
50%-chance blur σ 0.4–1.5, and gamma 0.7–1.4. Validation crops come from the val split
(disjoint seals), so training and validation share no seal.

## Measured results (verified on the val split)

- **Exact match 1,411/1,414 = 0.9979**, per-digit accuracy 0.9997, 0 localizer misses,
  3 recognizer errors.
- End to end: `main.py` processes images one at a time, but uses all CPU cores per
  image (~90 ms/image), dominated by PNG decode; well under a one-second-per-image line
  budget. The 7 digits of each seal are classified together in one batched SVM call.
- Held-out test split: exact match 1,412/1,414 = 0.9986.
- These are val/test numbers on the local splits; on an unseen, more-degraded test the
  organizers use, exact match may be lower.

## Reproduce the model

```bash
python harvest.py --input-dir <dataset>/train --labels <dataset>/splits/split_seals/train.csv --output-dir outputs/harvest_train --workers 4
python harvest.py --input-dir <dataset>/val   --labels <dataset>/splits/split_seals/val.csv   --output-dir outputs/harvest_val   --workers 4
python train.py --train outputs/harvest_train/harvest.csv --val outputs/harvest_val/harvest.csv
python eval.py  --input-dir <dataset>/val --labels <dataset>/splits/split_seals/val.csv --report outputs/eval_val.json
```

`outputs/` holds regenerable working data (harvested crops, reports) and is not part of
the shipped submission; only the code, `weights/svm.joblib`, and `requirements.txt` are.
# summer_school
