# Lightweight Hallucination Detector (Python)

This repo contains a lightweight hallucination detector for conversational
responses, plus a compact Low-Level Design (LLD) learning guide in Python.
The detector combines TF-IDF text features with small heuristic features so it
can run on CPU-only machines.

## Quickstart

1. Install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - `pip install -e .`
2. Run evaluation (trains and evaluates on the toy dataset):
   - `python3 scripts/evaluate.py`
3. Use the CLI:
   - `python -m halludetect.cli --response "Support lasts 12 months." --context "The plan includes 12 months of support." --json`

## Outputs

The project delivers:
- A classifier (TF-IDF + heuristics + logistic regression).
- Evaluation metrics in `reports/metrics.json`.
- Error analysis in `reports/error_analysis.md`.
- A CLI interface for ad-hoc detection.

## Latest evaluation results (toy test set)

- Accuracy: 0.58
- Precision: 0.57
- Recall: 0.67
- F1: 0.62

## Project structure

- `data/`: toy train/test CSV files.
- `docs/lld_guide.md`: LLD learning guide in Python.
- `reports/`: evaluation artifacts.
- `scripts/`: training and evaluation entrypoints.
- `src/halludetect/`: core package.

## Training and evaluation

Train only:
- `python3 scripts/train.py --train-path data/train.csv --model-path models/halludetect.joblib`

Train + evaluate:
- `python3 scripts/evaluate.py --train-path data/train.csv --test-path data/test.csv`

## CLI usage

Interactive:
- `python3 -m halludetect.cli`

Heuristic-only mode:
- `python3 -m halludetect.cli --response "The report shows 20% growth." --heuristic-only`

## LLD learning guide

Read `docs/lld_guide.md` for a quick guide on low-level design in Python.
