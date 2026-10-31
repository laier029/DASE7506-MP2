# Verifier models

Model C scores each candidate from its raw 1024-dimensional embedding. Model D first subtracts the mean embedding of the four candidates for the same question, then scores the relative embedding. Both use a deterministic ridge-regression correctness scorer trained only on the supplied training split.

Regularization was selected on validation from
`{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}`.
The selected values were `0.1` for Model C and `0.3` for Model D. The test split
was not used during model development.

Model E adds normalized final-answer frequency to Model D's score. A validation
grid from `0` to `2` in steps of `0.025` selected a consensus weight of `0.25`;
`0.30` produced the same validation accuracy, so the smaller weight was retained.

Model F adds a conservative arithmetic-consistency check. It examines only
numeric-only equations in LaTeX math fragments, skips unit-bearing and symbolic
expressions, and subtracts `0.325` when at least one checked equality is
inconsistent. The selected penalty begins a broad validation plateau rather than
being an isolated optimum.

Model G adds a final-answer/last-calculation consistency check to Model F. It
removes the declared-answer section, extracts the rightmost numeric result from
the last explicit equation (including plain-text equations), and subtracts
`0.075` only when that value conflicts with the parsed final answer. Missing or
unsupported calculations are neutral, and rounded integer answers are accepted.
The penalty was selected on the training split from `0` to `0.5` in steps of
`0.025`; `0.075` and `0.10` tied, so the smaller value was retained.

Model H evaluates distinct final-answer groups. Its group score is the best
Model G candidate score plus `0.30` times the group's mean candidate score, and
it returns the best candidate inside the winning group. This lets several
independently strong traces support the same answer without treating every
supporting trace as equally reliable.

Model I applies one protocol-level invariant to Model H: candidates without a
parseable final numeric answer are never selected. Such a candidate cannot earn
a point under the fixed scorer, while every benchmark group is guaranteed to
contain at least one correct (and therefore parseable) candidate.

Model J adds a broader arithmetic pass that can inspect equations containing
unit labels. It normalizes percentages and avoids comparing descriptive or
unit-conversion left-hand values. Errors found only by this broader pass receive
a smaller `0.075` penalty because the signal is less certain than Model F's
numeric-only check.

Model K adds two small within-question completeness features: log numeric-token
count with weight `0.01`, and log arithmetic-operator count with weight `0.02`.
Both are standardized across the four candidates, so the model rewards modestly
more worked reasoning without directly rewarding globally long answers. The
weights improved five-fold out-of-fold training predictions as well as the full
training and validation evaluations.

Model L adds a character `3`–`5`-gram TF-IDF `LinearSVC` with 25,000 features.
Its centered score is a small structure-dependent correction: `0` for `4–0`,
`-0.025` for `3–1`, `+0.025` for `2–2`, `-0.05` for `2–1–1`, and `-0.10` when
all answers differ. Five-fold out-of-fold training predictions selected these
directions; relative to a single `-0.05` weight, they preserve validation
accuracy while making substantially fewer unnecessary selection changes. Model
L requires the separately pinned dependencies in `requirements-model-l.txt`.

Model M keeps the shared 25,000-feature character vocabulary but fits separate
`LinearSVC` heads for the `3-1`, `2-2`, `2-1-1`, and all-distinct answer
structures. Its fixed text weights are `+0.025`, `-0.225`, `-0.10`, and
`-0.10`, respectively; `4-0` needs no text correction. Compared with Model L,
the fixed settings were non-negative on three additional random five-fold
partitions and improved validation from 748/798 to 749/798. Its uncompressed
inference assets total about 1.50 MiB.

Models N–T form the precision-first cascade. N rejects only answer groups whose
supporting traces share an unrepaired broad arithmetic error. O adds
calculation-quantity coverage; P rejects fractional answer groups; Q adds
occurrence-aware coverage; R adds a tightly gated omitted-factor switch; S adds
terminal target retention; and T adds the safe nonpositive-answer gate. Their
validation results progress from 750/798 (N) to 766/798 (T).

Models U–X add only five-fold-stable late signals and recoveries. U rewards
narrow implicit factors and softly penalizes broad arithmetic errors only in
`2-1-1`; V/W add conservative boxed-answer recovery; X rejects answer groups
whose explicit final declaration conflicts with the protocol-parsed answer.
Model X reaches 767/798.

Model Y adds a weak structure-aware lower-answer prior for remainder/difference
questions. Final Model Z adds terminal arithmetic-dependency coverage for
`now/currently/everything` targets and short-circuits questions with one unique
parseable answer group. It reaches 768/798 with five OOF deltas
`[10, 7, 8, 8, 5]` over X. The frozen entry point is `../submission.py`; see
`../EXPERIMENTS.md` and `../TRAINING.md` for the complete ablation, rejected
routes, exact checkpoint hashes and final commands.

Train from `code/`:

```bash
python train_embedding_model.py --representation raw --regularization 0.1 \
  --output runs/model_c_raw
python train_embedding_model.py --representation relative --regularization 0.3 \
  --output runs/model_d_relative
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --output runs/model_e_fusion
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --output runs/model_f_arithmetic
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --answer-step-penalty 0.075 --output runs/model_g_answer_step
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --answer-step-penalty 0.075 --group-mean-weight 0.30 \
  --output runs/model_h_group_aggregate
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --answer-step-penalty 0.075 --group-mean-weight 0.30 \
  --broad-arithmetic-penalty 0.075 --output runs/model_j_broad_arithmetic
python train_embedding_model.py --representation relative --regularization 0.3 \
  --consensus-weight 0.25 --arithmetic-penalty 0.325 \
  --answer-step-penalty 0.075 --group-mean-weight 0.30 \
  --broad-arithmetic-penalty 0.075 --number-count-weight 0.01 \
  --operation-count-weight 0.02 --output runs/model_k_reasoning_density
python train_text_fusion.py --base-checkpoint runs/model_k_reasoning_density \
  --output runs/model_l_text_fusion --folds 5 --text-weight -0.05 \
  --max-features 25000 --structure-aware
python train_structure_text_fusion.py \
  --base-checkpoint runs/model_k_reasoning_density \
  --output runs/model_m_structure_text --folds 5 \
  --max-features 25000 --c 0.1
```

Run validation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/embedding_linear.py \
  --checkpoint runs/model_c_raw --split validation \
  --output runs/model_c_raw/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_c_raw/validation-predictions.json \
  --output runs/model_c_raw/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/relative_embedding_linear.py \
  --checkpoint runs/model_d_relative --split validation \
  --output runs/model_d_relative/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_d_relative/validation-predictions.json \
  --output runs/model_d_relative/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/fused_relative_consensus.py \
  --checkpoint runs/model_e_fusion --split validation \
  --output runs/model_e_fusion/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_e_fusion/validation-predictions.json \
  --output runs/model_e_fusion/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/fused_arithmetic_verifier.py \
  --checkpoint runs/model_f_arithmetic --split validation \
  --output runs/model_f_arithmetic/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_f_arithmetic/validation-predictions.json \
  --output runs/model_f_arithmetic/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/fused_answer_step_verifier.py \
  --checkpoint runs/model_g_answer_step --split validation \
  --output runs/model_g_answer_step/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_g_answer_step/validation-predictions.json \
  --output runs/model_g_answer_step/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/group_aggregate_verifier.py \
  --checkpoint runs/model_h_group_aggregate --split validation \
  --output runs/model_h_group_aggregate/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_h_group_aggregate/validation-predictions.json \
  --output runs/model_h_group_aggregate/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/parseable_group_verifier.py \
  --checkpoint runs/model_h_group_aggregate --split validation \
  --output runs/model_i_parseable/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_i_parseable/validation-predictions.json \
  --output runs/model_i_parseable/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/broad_arithmetic_group_verifier.py \
  --checkpoint runs/model_j_broad_arithmetic --split validation \
  --output runs/model_j_broad_arithmetic/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_j_broad_arithmetic/validation-predictions.json \
  --output runs/model_j_broad_arithmetic/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/reasoning_density_verifier.py \
  --checkpoint runs/model_k_reasoning_density --split validation \
  --output runs/model_k_reasoning_density/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_k_reasoning_density/validation-predictions.json \
  --output runs/model_k_reasoning_density/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/text_fusion_verifier.py \
  --checkpoint runs/model_l_text_fusion --split validation \
  --output runs/model_l_text_fusion/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_l_text_fusion/validation-predictions.json \
  --output runs/model_l_text_fusion/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method models/structure_text_fusion_verifier.py \
  --checkpoint runs/model_m_structure_text --split validation \
  --output runs/model_m_structure_text/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/model_m_structure_text/validation-predictions.json \
  --output runs/model_m_structure_text/validation-score.json
```
