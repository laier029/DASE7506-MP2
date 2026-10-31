# Frozen Model Z checkpoint

Pass this directory to `predict.py` with
`--checkpoint runs/model_z_robust_semantic`.

Required inference assets:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.npz` | 15,473 | `4d9e50ace57b75d744bf37965288754339457f22f17ca955143a67a0d3999cb6` |
| `structure_text_model.joblib` | 1,559,531 | `9cd78234fc5d8b22819df9b2d843739bc2c6077d2ebc8695312d0e3102e11586` |

Total uncompressed inference assets: **1,575,004 bytes**. The other files in
this directory are metadata, frozen predictions, the official score, timing,
source/asset manifests and audit evidence; they are not additional inference
weights.

`test-evaluation.json` records the one frozen test prediction and one scoring
run: **384/467 = 82.22698072805139%**. The matching source version is the Git
tag `v1.0-model-z`. A convenience archive containing the two assets, this
README, metadata, source manifest and evaluation summary is published as
`release/model-z-checkpoint-v1.0.zip`.
