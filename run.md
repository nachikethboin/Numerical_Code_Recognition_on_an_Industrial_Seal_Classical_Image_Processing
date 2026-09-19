# Run guide

Classical seal-number OCR. The trained model ships in `weights/svm.joblib`, so the
run commands work immediately — no training needed.

**Environment:** the project virtual environment already exists at `.venv`
(Python 3.12 with numpy, opencv, scikit-learn, scipy, joblib). Run every command
below from the project root `C:\summerschool`.

Fresh machine only (deps not installed yet):

```
.venv/Scripts/python.exe -m pip install -r mywork/requirements.txt
```

---

## 1. Run on a folder of seal PNGs → write `<team>.csv`

This is the competition entry point. Replace `<INPUT_FOLDER>` with the folder of PNGs
and `<OUTPUT_FOLDER>` with where the CSV should go.

```
.venv/Scripts/python.exe mywork/main.py --input-dir <INPUT_FOLDER> --output-dir <OUTPUT_FOLDER> --team dbc_lab
```

Output: `<OUTPUT_FOLDER>/dbc_lab.csv` with header `filename;number`. The log line
prints the total processing time and ms/image.

## 2. Check accuracy against a ground-truth CSV

Needs a labels CSV in `filename;number` format (e.g. the organizers' `gt.csv`).

```
.venv/Scripts/python.exe mywork/eval.py --input-dir <INPUT_FOLDER> --labels <GT_CSV> 
```

Prints exact-match rate, per-digit accuracy, localizer misses, and recognizer errors.
`errors = seals - exact_match`.

## 3. Compute the competition score (euros; lower is better)

Feed the error count (from step 2) and the processing time in seconds (from step 1)
into the organizers' `test_example/score.py`. Replace the `LineConfig` values and the
two numbers with the official ones.

```
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'test_example'); from score import compute_score, LineConfig; print(compute_score(errors=0, processing_time_s=1.25, config=LineConfig(plombs_per_hour=3600, eur_per_minute=0.1, technician_fee=1, downtime_per_error=2)))"
```

---

## Optional — rebuild the model from scratch

Only needed to reproduce `weights/svm.joblib`; requires the full dataset under
`ai_summer_school_dataset/`.

```
.venv/Scripts/python.exe mywork/harvest.py --input-dir ai_summer_school_dataset/train --labels ai_summer_school_dataset/splits/split_seals/train.csv --output-dir mywork/outputs/harvest_train --workers 4
```
```
.venv/Scripts/python.exe mywork/harvest.py --input-dir ai_summer_school_dataset/val --labels ai_summer_school_dataset/splits/split_seals/val.csv --output-dir mywork/outputs/harvest_val --workers 4
```
```
.venv/Scripts/python.exe mywork/train.py --train mywork/outputs/harvest_train/harvest.csv --val mywork/outputs/harvest_val/harvest.csv
```
