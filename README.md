# Peptide Property Benchmark

This repository contains the training and evaluation code used for peptide
property prediction experiments. It covers traditional machine-learning
baselines, deep learning models, and frozen protein language-model
embeddings. Both random and similarity-aware dataset splits are supported.

## Repository layout

```text
.
├── pephub/
│   ├── raw_data/          # Input datasets used by all experiments
│   ├── dataset.py         # Dataset loading and validation
│   ├── featurizer.py      # Peptide feature extraction
│   ├── splitter.py        # Random and similarity-aware splitting
│   └── results.py         # Shared metrics JSON schema
├── tests/                 # Unit tests for reusable data utilities
├── ml_train.py            # Random Forest, SVM/SVR, and XGBoost
├── dl_train.py            # LSTM and Transformer sequence models
├── esm_train.py           # Frozen ESM encoder with an MLP head
├── protbert_train.py      # Frozen ProtBERT encoder with an MLP head
└── run_*.sh               # Reproducible random/similarity experiment runners
```

Generated checkpoints, metrics, run state, logs, and downloaded pretrained
weights are intentionally excluded from version control.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Similarity-aware splitting additionally requires
[MMseqs2](https://github.com/soedinglab/MMseqs2) on `PATH`.

## Pretrained encoders

Large pretrained weights are not stored in this repository because the model
files exceed GitHub's normal file-size limit. Download compatible Hugging Face
model snapshots into these directories:

```text
pretrained/esm2_t12_35M_UR50D/
pretrained/prot_bert/
```

The expected upstream models are `facebook/esm2_t12_35M_UR50D` and
`Rostlab/prot_bert`. The shell runners pass the local directories explicitly,
so model loading behavior is unchanged.

## Data format

Every CSV file under `pephub/raw_data/` must contain exactly these columns:

```csv
id,peps,label
1,ACDEFGHIK,1
2,LMNPQRSTV,0
```

Datasets whose filename contains `reg` are treated as regression tasks; all
other datasets are treated as classification tasks. This naming convention is
part of the original experiment logic.

## Running experiments

Each runner executes both random and similarity-aware splits with seeds 42,
43, and 44:

```bash
bash run_ml.sh
bash run_dl.sh
bash run_esm.sh
bash run_protbert.sh
```

To run a subset, invoke the Python entry point directly. For example:

```bash
python ml_train.py --split_method random --datasets AMP
python dl_train.py --split_method random --models lstm --datasets AMP
python esm_train.py --split_method random --datasets AMP
python protbert_train.py --split_methods random --datasets AMP
```

Use `--help` on any entry point for the complete set of runtime options.

## Model configurations

The model configurations used by the experiment scripts are:

| Model | Parameters |
|---|---|
| Random Forest | 300 trees, unlimited depth |
| SVM | linear kernel, `C=1.0` |
| SVR | RBF kernel, `C=10.0`, `epsilon=0.1` |
| XGBoost | 300 trees, depth 6, learning rate 0.1 |
| LSTM | embedding 50, hidden size 256, 2 layers |
| Transformer | model size 64, 4 heads, 2 layers, FFN ratio 4 |
| ESM + MLP | head learning rate 0.001, hidden sizes 256 and 128 |
| ProtBERT + MLP | head learning rate 0.001, hidden sizes 256 and 128 |

## Outputs

All generated artifacts are written under `outputs/`:

```text
outputs/
├── checkpoints/           # Resume checkpoints
├── metrics/               # Final metrics reports
├── models/                # Fitted model files
└── <model-family>/         # Internal resumable run state and summaries
```

Every final metrics file uses schema version `1.0` and the same top-level
fields: `dataset`, `task_type`, `split`, `model`, `parameters`, `selection`,
and `test`. Model checkpoint formats remain model-specific.

## Validation

```bash
python -m compileall -q .
python -m pytest -q
```

## License

This project is released under the terms in [LICENSE](LICENSE).
