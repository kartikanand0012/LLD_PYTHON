# Low-Level Design (LLD) in Python

LLD focuses on how classes, data structures, and functions collaborate to
implement a feature. It is the bridge between requirements and code. This guide
is short and practical so you can map concepts directly onto the codebase in
this repo.

## LLD mindset

1. **Identify the core entities**
   - What nouns show up in the problem statement?
   - Example: response, context, detector, model, feature, dataset.
2. **Define responsibilities**
   - Each class or module should have a single reason to change.
   - Example: `features.py` owns feature extraction only.
3. **Decide collaborations**
   - Which component calls which?
   - Example: `scripts/evaluate.py` orchestrates loading data, training, and
     evaluation.
4. **Model data flow**
   - What data structures move between modules?
   - Example: `Sample` objects move through dataset -> model -> evaluation.

## Core LLD principles used

### Single Responsibility
- `dataset.py` only loads and structures data.
- `features.py` only creates model-ready features.
- `model.py` only trains, saves, and predicts.
- `evaluate.py` computes metrics and error summaries.

### Open/Closed
- New heuristics can be added in `heuristics.py` without changing the training
  code.

### Dependency Direction
- Higher-level scripts (`scripts/*.py`) depend on low-level modules; core logic
  does not depend on scripts.

## LLD for this detector (component view)

```
CLI / scripts
   |
   v
dataset.py -> features.py -> model.py -> evaluate.py
                   |
                   v
            heuristics.py (rules)
```

## Example design sketch

### Class: `Sample`
- **Fields**: `response`, `context`, `label`
- **Reason to change**: dataset schema changes

### Module: `features.py`
- `select_text(samples)` builds text input for TF-IDF.
- `extract_heuristics(samples)` builds numeric rule features.
- **Reason to change**: add/remove features.

### Module: `model.py`
- `build_model()` wires a vectorizer and classifier.
- `train_model()` handles fitting.
- **Reason to change**: change the ML model or training strategy.

### Module: `heuristics.py`
- `heuristic_judgement()` is a fallback rule-based detector.
- **Reason to change**: add detection rules.

## How to apply LLD to new features

1. **Write a short problem statement** (2 to 4 lines).
2. **List candidate classes/modules** and the one responsibility each owns.
3. **Sketch interactions** (a small diagram or bullet flow).
4. **Define data contracts** (what each function returns and expects).
5. **Implement with thin orchestration** and reuse core modules.

## Example exercise

Problem: "Add a confidence explanation for the model."

LLD sketch:
- `explain.py` module: generates explanation from features.
- Update CLI to call `explain.py` when `--explain` is passed.
- Keep `model.py` unchanged to preserve responsibilities.
