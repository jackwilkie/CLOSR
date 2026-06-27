# Baseline Implementations

This directory contains reference implementations of the baseline models reported in the paper. They are provided for comparison and inspection, separate from the main CLAD and CLOSR training/evaluation scripts at the repository root.

The main paper methods are still trained and evaluated with:

```bash
python3 train_clad.py
python3 eval_clad.py
python3 train_closr.py
python3 eval_closr.py
```

The files here are intended as reusable baseline components rather than a single unified experiment runner.

## Structure

```text
baseline_implementations/
  binary_classification/
    autoencoder.py
    daelr.py
    duad_train.py
    isolation_forrest.py
    renoir.py
    svdd.py
    svm.py
  openset_recognition/
    crosr.py
    doc.py
    multistage_nids.py
    openmax.py
    siamese_network.py
  common/
    feed_forward.py
    losses.py
    metrics.py
    model_eval.py
    open_auc.py
    process_batch.py
    training_loops.py
```

## Binary Classification Baselines

These baselines are used for anomaly/binary intrusion detection comparisons:

- Autoencoder: `binary_classification/autoencoder.py`
- DAE-LR / DUAD-LR components: `binary_classification/daelr.py`
- DUAD: `binary_classification/duad_train.py`
- Isolation Forest: `binary_classification/isolation_forrest.py`
- RENOIR: `binary_classification/renoir.py`
- Deep SVDD: `binary_classification/svdd.py`
- One-class SVM: `binary_classification/svm.py`

Example:

```python
from baseline_implementations.binary_classification.svm import fit_svm_auroc

results = fit_svm_auroc(
    x_train=x_train,
    x_val=x_val,
    y_val=y_val,
    y_train=y_train,
)
```

## Open-Set Recognition Baselines

These baselines are used for open-set recognition comparisons:

- CROSR: `openset_recognition/crosr.py`
- DOC: `openset_recognition/doc.py`
- Multi-stage NIDS: `openset_recognition/multistage_nids.py`
- OpenMax: `openset_recognition/openmax.py`
- Siamese Network: `openset_recognition/siamese_network.py`

## Shared Utilities

The `common/` directory contains helper code used by the baseline implementations, including metric calculation, OpenAUC, batch processing, shared losses, feed-forward network blocks, and legacy-style training utilities.

These helpers are baseline-local on purpose. They avoid adding old project-specific packages such as `utils`, `model_training`, `metric_learning`, `factory`, or `data_read` to the top level of this repository.

## Dependencies

Install the repository dependencies with:

```bash
pip install -r requirements.txt
```

Some open-set baselines use `libmr`, which is included in `requirements.txt`.