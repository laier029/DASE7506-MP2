# DASE7506 MP2 — Model Z Submission

This is the frozen submission repository for student **3036779621**. Model Z
selects one of four LLM-generated solutions with a precision-first verifier:
high-confidence negative gates remove demonstrably bad answer groups, then a
learned and structured scorer ranks the survivors.

## Frozen result

- Protocol: `dase7506-mp2-open-hard-verifier-v5`
- Validation: **768 / 798 = 96.2406015037594%**
- Test: **384 / 467 = 82.22698072805139%**
- Test prediction runs: **1**
- Test scoring runs: **1**
- Frozen release tag: **`v1.0-model-z`**
- Inference assets: **1,575,004 bytes uncompressed**

The test result was observed only after the source and checkpoint were frozen.
No method, threshold or checkpoint was changed in response to the test score.

## Submission artifacts

- [Final report (PDF)](code/report/DASE7506_MP2_Model_Z_Report.pdf)
- [Editable report source (DOCX)](code/report/DASE7506_MP2_Model_Z_Report.docx)
- [Frozen entry point](code/submission.py)
- [Checkpoint and audit record](code/runs/model_z_robust_semantic/)
- [Downloadable checkpoint bundle](release/model-z-checkpoint-v1.0.zip)
- [Official test evaluation JSON](code/runs/model_z_robust_semantic/test-evaluation.json)
- [Training and reproduction instructions](code/TRAINING.md)
- [Experiment and ablation record](code/EXPERIMENTS.md)
- [Course guide](guide/GUIDE.md)

All supplied data, licenses and attribution files are retained. The original
starter `code/PACKAGE_MANIFEST.json` is preserved as provenance; because
`submission.py` is the required student implementation, its original starter
hash is not expected to match. The frozen inference-source and checkpoint
hashes are recorded in
`code/runs/model_z_robust_semantic/source-manifest.json`.

## Quick reproduction

```bash
git clone https://github.com/laier029/dase7506-mp2.git
cd dase7506-mp2/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py \
  --checkpoint runs/model_z_robust_semantic \
  --split validation \
  --output runs/model_z_robust_semantic/reproduced-validation-predictions.json

python score_predictions.py --split validation \
  --predictions runs/model_z_robust_semantic/reproduced-validation-predictions.json \
  --output runs/model_z_robust_semantic/reproduced-validation-score.json
```

Expected validation result: `768 / 798 = 96.2406015037594%`.
The frozen test output is supplied for reproduction; it should not be used for
further method selection.

## AI and reuse disclosure

OpenAI Codex provided substantive assistance with project interpretation,
local environment checks, verifier brainstorming, Python code and test
drafting, experiment orchestration, debugging, result summarisation, and
report structure and language. I proposed the main two-stage research
direction, decided whether experiments should continue, and made the final
decision to submit Model Z. Suggestions were accepted only after local runs,
OOF analysis, validation checks, regression tests and resource verification.

The frozen verifier does not call Codex, ChatGPT or any external LLM/API during
inference. No external data, answer lookups or unapproved pretrained weights
were introduced. I retain responsibility for the implementation,
interpretation and submission. Questions and references derive from GSM8K
under its retained MIT licence; candidate solutions and fixed representations
were supplied by the course and prepared with Qwen3-0.6B.
