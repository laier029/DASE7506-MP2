# Model Z Training and Reproduction

Run every command from `code/` with Python 3.12. The frozen checkpoint is
already included in `runs/model_z_robust_semantic/`; retraining is not required
for evaluation.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The measured environment used Python 3.12.10, NumPy 2.5.3,
scikit-learn 1.9.1, joblib 1.6.0, SciPy 1.18.1 and threadpoolctl 3.7.0.
The direct dependencies are pinned in `requirements.txt`; the full measured
runtime versions are also recorded in the checkpoint `metadata.json`.

## Rebuild the learned assets from training data

The first command fits the relative-embedding ridge scorer and stores all
fixed reasoning-density settings. The second command fits the four
answer-structure-specific character TF-IDF/LinearSVC heads using five
question-level folds. Both scripts read training references only.

```bash
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --answer-step-penalty 0.075 --group-mean-weight 0.30 \
  --broad-arithmetic-penalty 0.075 --number-count-weight 0.01 \
  --operation-count-weight 0.02 --output runs/rebuilt_model_k

python train_structure_text_fusion.py \
  --base-checkpoint runs/rebuilt_model_k \
  --output runs/rebuilt_model_z_assets --folds 5 \
  --max-features 25000 --c 0.1
```

`LinearSVC` uses `random_state=7506`. The deterministic fold assignment is
`question_index % 5`. Model Z's later filters and survivor signals are
training-free fixed code. Their weights were accepted only after five OOF
schemes and validation checks documented in `EXPERIMENTS.md`.

The robustness audit uses the deterministic modulo split plus four balanced
random five-fold assignments with seeds `11`, `29`, `47` and `83`. The frozen
audit output is `runs/model_z_robust_semantic/oof-audit.json`.

## Evaluate the frozen checkpoint

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py \
  --checkpoint runs/model_z_robust_semantic \
  --split validation \
  --output runs/model_z_robust_semantic/submission-validation-predictions.json

python score_predictions.py --split validation \
  --predictions runs/model_z_robust_semantic/submission-validation-predictions.json \
  --output runs/model_z_robust_semantic/submission-validation-score.json
```

Expected validation result: `768 / 798 = 96.2406015037594%`.

To verify rebuilt assets without replacing the frozen checkpoint, use
`--checkpoint runs/rebuilt_model_z_assets`. Do not compare raw prediction JSON
hashes because timing fields vary; compare the `selected` payload instead.

A final clean temporary rebuild reproduced both asset hashes byte-for-byte,
the complete selected payload hash, and 768/798 validation. The machine-readable
record is `runs/model_z_robust_semantic/rebuild-audit.json`.

## Frozen asset hashes

```text
model.npz
4d9e50ace57b75d744bf37965288754339457f22f17ca955143a67a0d3999cb6

structure_text_model.joblib
9cd78234fc5d8b22819df9b2d843739bc2c6077d2ebc8695312d0e3102e11586
```

The two files total 1,575,004 bytes uncompressed. Test references were not used
for training, tuning, selection or the recorded validation analysis. After the
freeze, one official test run scored 384/467 = 82.22698072805139%; see
`test-freeze.json` and `test-evaluation.json`. No further tuning is permitted.

## Freeze and verify the source bundle

Generate the transitive local-source and inference-asset hash manifest after
the final code freeze:

```bash
python experiments/source_manifest.py \
  --output runs/model_z_robust_semantic/source-manifest.json
```

The current manifest contains 27 local inference-source files and the two
checkpoint assets. Regenerate it after any source or asset change; a changed
hash means the frozen bundle must be revalidated before test scoring.

Before the one-time test run, verify the bundle without rewriting the recorded
manifest:

```bash
python experiments/source_manifest.py --verify
python experiments/pretest_freeze_audit.py
```

The second command checks the recorded validation/OOF results, prediction
schema and determinism, every inference-source and asset hash, fresh-rebuild
evidence, original runner/parser/scorer and train/validation data hashes, the
16 MiB asset limit, and confirms that no test prediction or score output
exists. It does not open test inputs or references. That pre-test audit has
already been recorded with 128 passing checks. Now that the test outputs exist,
it is expected to fail its "no test outputs" assertion; do not overwrite the
archived pre-test audit. `source_manifest.py --verify` remains valid after test.
