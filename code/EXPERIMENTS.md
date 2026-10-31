# MP2 Experiment Record

This record describes the frozen development state for protocol
`dase7506-mp2-open-hard-verifier-v5`. Training references were used for fitting
and question-level out-of-fold (OOF) checks. Validation was used for method
development. Test references and scores remained sealed throughout development;
after the source and checkpoint were frozen, one test prediction and one score
run produced 384/467 = 82.22698072805139%.

## Research question

Can a verifier improve over learned candidate scoring by separating two jobs?

1. **Negative filtering:** reject an answer group only when an error signal has
   very high empirical precision.
2. **Survivor scoring and recovery:** rank the remaining groups using learned
   and structured signals, then apply narrowly gated corrections with explicit
   safe fallbacks.

The expected benefit was fewer convincing-but-impossible selections without
allowing noisy arithmetic heuristics to remove correct answers.

## Feature promotion rule

Development used the following admission order rather than accepting any
validation gain:

1. Define a question-conditioned mechanism that does not use IDs or references
   at inference.
2. For an irreversible gate, measure precision at the answer-group level and
   provide an all-groups-removed fallback.
3. Select a bounded weight/threshold on the primary OOF scheme, then require
   non-negative behavior across four additional balanced five-fold schemes.
4. Inspect fixes, harms and answer-group changes, not only the net correct
   count. Large offsetting churn is treated as instability.
5. Require no validation regression, then rerun the frozen production functions
   rather than relying on a simplified probe implementation.
6. Recheck arbitrary-bank alignment, candidate-order invariance, determinism,
   source/asset hashes and resource limits before promotion.

This rule explains why the validation-neutral remainder feature was accepted
(consistent OOF evidence and bounded effect), while the later ordinary-total
dependency feature was rejected despite non-negative net OOF deltas (its fixes
and harms offset within folds and its direction lacked a clear mechanism).

## Current validation ablation

All rows below were rerun after the currency/LaTeX parser correction, using the
same 798 validation questions.

| Model | Added component | Correct | Accuracy |
|---|---|---:|---:|
| M | structure-specific embedding + character text scorer | 749 | 93.8596% |
| N | unanimous broad-arithmetic group filter | 750 | 93.9850% |
| O | calculation unique-quantity coverage | 757 | 94.8622% |
| P | fractional-answer negative filter | 761 | 95.3634% |
| Q | occurrence-aware calculation coverage | 763 | 95.6140% |
| R | tightly gated omitted-factor switch | 765 | 95.8647% |
| S | terminal target-retention score | 765 | 95.8647% |
| T | safe nonpositive filter and complete initial cascade | 766 | 95.9900% |
| U | stable implicit-factor reward + scoped broad penalty | 766 | 95.9900% |
| V | selected-candidate boxed-answer redirect | 767 | 96.1153% |
| W | consistent boxed-answer endorsement | 767 | 96.1153% |
| X | explicit final-declaration hard filter + W recovery | 767 | 96.1153% |
| Y | conservative remainder/difference answer-rank prior | 767 | 96.1153% |
| **Z** | cumulative terminal-dependency coverage + Y | **768** | **96.2406%** |

The frozen submission is Model Z. Model U did not alter a validation answer
group, but improved all five OOF schemes by `[4, 3, 5, 7, 6]` correct
questions. Model X's late filter/recovery stages improved Model U by
`[6, 5, 5, 6, 5]`; total OOF improvements over Model T were
`[10, 8, 10, 13, 11]`. This was retained because every OOF scheme improved and
validation did not regress. Model Y improved Model X by `[0, 4, 2, 1, 1]`
across the five OOF schemes. An initial uniform dependency weight `+0.40`
improved Model X by `[6, 7, 6, 5, 4]` and fixed validation question 617
without a harm. A global `+0.50` introduced one validation harm, while a
global `+0.60` regressed two OOF schemes.

The final structure-aware settings use dependency weight `+0.50` for `3-1`
and `+0.40` otherwise, plus remainder-rank weight `-0.10` for `2-2` and
`2-1-1` and `-0.025` otherwise. With these bounded adjustments, Model Z's
final OOF deltas over X are `[10, 7, 8, 8, 5]`; fixes are
`[12, 12, 14, 10, 10]` and harms are `[2, 5, 6, 2, 5]`.

Before any learned scoring, Model Z short-circuits questions with exactly one
parseable answer group. This covered 2,641/4,755 train and 443/798 validation
questions, with 100% correctness on both development splits. The remaining
2,114 train and 355 validation questions run through the complete verifier.

## Precision-first gates

The following statistics are empirical development evidence, not universal
guarantees:

- Fractional final-answer groups: 279/279 wrong on train and 56/56 wrong on
  validation.
- Safe nonpositive groups (no explicit negative question input): 122/122 wrong
  on train and 25/25 wrong on validation.
- Unanimous shared broad-arithmetic mismatches removed no correct group in the
  five OOF checks or validation audit.
- Explicit `Final Answer:` conflict groups: 25/25 wrong on train and 5/5 wrong
  on validation.
- Protocol/boxed conflicts: all 25 flagged train candidates and all 8 flagged
  validation candidates were wrong under the official parser. When a visible
  boxed target group existed, it was correct in 16/17 train cases and 4/4
  validation cases.

Finite-sample uncertainty differs substantially across gates. With zero
observed false rejections, the one-sided 95% exact lower bound on wrong-group
precision was 98.93%/94.79% for the fractional gate (train/validation),
97.57%/88.71% for safe nonpositive answers, 88.71%/54.93% for explicit final
declaration conflicts, and only 60.70%/36.84% for the six/three broad-arithmetic
groups. The arithmetic gate is therefore supported more by its narrow logical
definition and consistent marginal result than by sample size alone.

Inside the complete Model Z selector, all four hard gates together improved
the five OOF schemes by `[21, 20, 18, 20, 19]` correct questions and validation
by 7: the same final scorer, switches and recovery path scores 761/798 without
the four gates and 768/798 with them. Leave-one-gate-out contributions were
`[1, 1, 1, 1, 1]`/+1 validation
for broad arithmetic, `[13, 12, 10, 11, 9]`/+4 for fractional answers,
`[5, 5, 5, 6, 7]`/+2 for safe nonpositive answers, and
`[2, 2, 2, 2, 2]`/+0 for final-declaration conflicts. Thus a gate may have
perfect flag precision yet no validation selection effect because the scorer
already avoids its flagged groups. Full results are saved in
`gate-precision-audit.json` and `hard-gate-marginal-audit.json`.

The joint hard-gate gain is concentrated exactly where answers disagree. It
has no effect on the 4-same structure. Across the five OOF schemes, correct
contributions are `+5..+6` for `3-1`, `+3` for `2-2`, `+7..+10` for `2-1-1`
and `+2..+4` for all-distinct, with no correct-to-wrong harm in any structure.
The corresponding validation gains are +1, +2, +1 and +3. This supports both
the unique-answer fast path and the decision to reserve detailed verification
for disagreement rows. See `hard-gate-structure-audit.json`.

Every hard-filter stage restores all parseable groups if filtering would leave
none. Boxed-answer recovery preserves the scored choice if targets are absent,
conflicting or unavailable as a protocol-visible candidate group.

The combined hard filters did not naturally remove every group in any of the
4,755 train or 798 validation questions, so that safety branch is verified by
an explicit synthetic regression test. Boxed safe fallbacks did have natural
support: train contained 2 conflicting-target and 6 protocol-invisible-target
questions; validation contained 4 protocol-invisible-target questions. Both
conflicting and invisible cases also have direct regression tests.

## Rejected or bounded experiments

- Individual arithmetic-error rejection was too noisy; only group-unanimous
  signatures were safe.
- Plain-text arithmetic penalties improved several OOF schemes but reduced
  validation by two questions.
- Step-mismatch and question-number hard filters had insufficient wrong-group
  precision.
- General question/candidate lexical overlap, answer support and cross-candidate
  intermediate agreement selected zero or unstable fusion weights.
- Word TF-IDF, pairwise embedding ranking, hard-only embedding training,
  PCA/kNN and nonlinear static meta-rankers did not improve the accepted base.
- Question-only majority/minority priors and sparse question-by-candidate-state
  interactions selected zero weights.
- Generic target-operation rules, answer-magnitude priors, complement rules and
  occurrence-based additive switches were fold-unstable or validation-negative.
- `2-1-1` minority fallback reduced every OOF result; `3-1` majority-to-singleton
  switching selected a zero threshold and changed nothing.
- Broad `Final Answer:` endorsement was rejected: despite a 99.29% wrong-candidate
  signal, its proposed target group was only 89.02% correct on train, producing
  large OOF losses. Model X uses it only as a negative all-group filter.
- Area/volume mismatch fixed one validation error but had no training support,
  so it was not promoted from a diagnostic rule.
- Explicit final-unit mismatch had only 71.23% wrong-group precision on train
  and reduced all five OOF results as a hard gate. Its soft version selected
  weight `-0.20` but had OOF deltas `[1, 2, 0, 2, -1]` and validation `-1`.
- Weeknight/weekend and dozen factor extensions were fold-unstable or selected
  the opposite sign; simple numeric occurrence did not establish correct use.
- Severe plain-text arithmetic filtering fixed one validation error but had
  only 42/58 wrong-group precision and reduced four of five OOF results.
- Learned group-mean and group-mean-plus-dispersion embedding representations
  selected zero fusion weight.
- Treating cumulative dependency as a hard gate was rejected: even the least
  aggressive gate had only 69/83 wrong-group precision and reduced every OOF
  result by 9–10 questions. The same evidence works only as a soft score.
- Dependency complexity for remainder/difference targets was fold-unstable;
  both positive and negative weights caused OOF regressions.
- Structure-specific dependency coverage for ordinary `total/altogether`
  targets did not justify another feature. A `3-1` weight of `-0.10` had net
  OOF deltas `[1, 0, 1, 0, 1]` and no validation answer-group change, but two
  folds merely exchanged 3 fixes for 3 harms and 2 fixes for 2 harms. The
  counterintuitive sign, hidden churn and zero validation effect made the net
  result too fragile. Other structures were fold-unstable or negative.
- Replacing hard gates with candidate-score penalties from a fixed `0.1–4.0`
  grid did not produce a smaller, consistently better alternative. Weights
  were selected on the primary OOF scheme; the other four schemes were used
  only for robustness checks. Fractional and nonpositive gates needed weight
  `4.0` to match hard-gate correct-count contributions; final-declaration
  conflicts needed `2.0`; broad arithmetic needed `1.0` to match OOF counts
  but `2.0` to recover its validation gain. The hard version remains simpler
  and has an explicit all-groups-removed fallback. Full curves are in
  `gate-soft-vs-hard.json`.

These failures support the main conclusion: arithmetic or formatting evidence
is useful as a conservative gate, while semantic correctness still requires a
learned survivor scorer. More aggressive rule coverage quickly overfits.

## Error and uncertainty analysis

Model Z accuracy by final-answer structure is:

| Structure | Correct | Total | Accuracy |
|---|---:|---:|---:|
| 4 | 379 | 379 | 100.00% |
| 3-1 | 197 | 204 | 96.57% |
| 2-2 | 41 | 45 | 91.11% |
| 2-1-1 | 117 | 130 | 90.00% |
| 1-1-1-1 | 34 | 40 | 85.00% |

This supports skipping extra analysis for unanimous answers and concentrating
computation on disagreement structures. Group-score margin is informative but
not perfectly calibrated: among the disagreement rows with a defined final
group margin, selections with margin 0–0.5 are 22/30 correct (73.33%), while
margins at least 2 are 158/165 correct (95.76%). The three negative-margin
choices are deliberate gated recoveries and are all correct, so raw margin
alone should not override the cascade.

Relative to Model M, Model Z fixes 20 validation questions and harms 1, for a
net gain of 19. A Wilson 95% interval for the final 768/798 accuracy is roughly
94.68%–97.35%; the interval describes sampling uncertainty on this bank and was
not a prediction of the later post-freeze test score.

## Post-freeze test evaluation

The frozen Model Z was run on test exactly once, followed by one independent
scorer run. It achieved **384/467 = 82.22698072805139%**. Prediction took 1.83
seconds internally and 1.94 seconds wall time on the development Mac; scoring
took 0.010 seconds internally. The entry point and both asset hashes were
verified unchanged after evaluation. The lower test accuracy is consistent
with the protocol difference that all test rows contain answer disagreement,
whereas validation includes 443 unique-answer fast-path rows. However, removing
those easy rows still leaves validation disagreement accuracy at
325/355 = 91.5493%, 9.32 percentage points above test. Thus the missing fast
path explains only part of the gap; the test disagreement bank is also harder
or differently distributed. This is descriptive analysis, not a basis for
post-test tuning.

Frozen inputs, outputs, hashes and the no-further-tuning declaration are in
`test-freeze.json` and `test-evaluation.json`.

## Post-freeze train-only router research

An isolated Model Z2 experiment tested whether multiple small routers could
learn when three independent answer-group experts should override Model Z.
The search used five train OOF partitions and fixed one gate before five new
OOF audit partitions were generated. The routers learned fix-versus-harm
directly, required at least five of six router votes, required all three
experts to agree, and operated only on `2-1-1` and `1-1-1-1` questions with a
low enough Model Z group margin. No test input, reference or score was opened.

The frozen gate was positive on every development OOF partition
(`[+8, +6, +7, +14, +8]`) and every new audit partition
(`[+14, +7, +9, +8, +14]`). Across the fresh audit it made 103 fixes and 51
harms; its pooled effective-switch precision was 66.88%, with a one-sided 95%
Wilson lower bound of 60.40%. Despite that train-side stability, the single
frozen validation check fell from 768/798 to 766/798: four switches produced
one fix and three harms. The candidate was therefore rejected under the
predeclared no-validation-regression rule. Model Z remains the submission
model, and the gate was not tuned again after seeing this validation result.
The full protocol and results are in `oof-router-ensemble-z2.json`.

One final frozen adjustment then replaced the single-seed router training set
with five pooled train OOF representations. Repeated labelled representations
of the same question received total weight one; the experts, allowed
structures, five-of-six vote rule, probability threshold and Model Z margin
gate were unchanged. Five additional audit seeds, unused by either earlier
router experiment, were all positive: `[+9, +8, +5, +10, +13]`. Together they
made 92 fixes and 47 harms, for a net gain of 45 and a one-sided 95% Wilson
precision lower bound of 59.33%.

The final validation check returned to 768/798 rather than exceeding it: four
switches produced two fixes and two harms. Thus pooling improved the rejected
router by two validation answers but made no net improvement over Model Z.
It remains a research candidate only and was not productionized or substituted
into `submission.py`. No further threshold or feature adjustment was made
after this check. Full results are in `oof-router-pooled-final.json`.

An exact paired two-sided binomial test is descriptive only because validation
was used for development: M-to-Z's 20 fixes versus 1 harm gives
`p = 2.10e-5`, while X-to-Z's single fix and zero harms gives `p = 1.0`.
Therefore the last one-question gain should not be presented as strong
standalone statistical evidence; its five-scheme OOF stability is the main
reason for retaining the soft feature.

The remaining 30 validation errors are dominated by semantic composition:
missing a baseline stock, mishandling "more than" or percent wording, omitting
people/trips/time periods, or solving an intermediate quantity instead of the
asked total. These cases explain why broader arithmetic penalties and simple
majority fallbacks stopped helping.

A manual, exclusive diagnostic grouping of those 30 errors gives 11
multiplicity/time/participant-accounting cases, 9 state-transition or baseline
ledger cases, 8 relational-algebra/comparison cases, 1 discrete-remainder case
and 1 geometry case. This taxonomy is interpretation of validation errors, not
an independently estimated test distribution. It suggests that the next
meaningful model family would explicitly represent entities, state changes and
relations; adding more isolated keyword penalties would mostly repeat the
unstable experiments above. A compact question/answer/tail catalog is saved as
`validation-error-catalog.json` for report examples and future analysis.
The corresponding manual row-index mapping is in
`validation-error-taxonomy.json`.

## Frozen resources

- Method: `submission.py`
- Checkpoint: `runs/model_z_robust_semantic/`
- Assets: 1,575,004 bytes uncompressed (about 1.50 MiB)
- Four-thread submission validation prediction: 1.48 seconds internal, 1.57 seconds wall
  clock on the development Mac
- Peak process RSS during that run: 223,756,288 bytes
- Reference-free stress run on all 4,755 training inputs (2,114 full-verifier
  rows): 6.24 seconds internal, 6.34 seconds wall and 598,343,680 bytes peak
  RSS. This larger input-only run gives a less optimistic resource check than
  validation's 443-row fast-path coverage; it is not a test-score estimate.
- Independent scoring of that 4,755-row stress output took 0.094 seconds
  internally and 0.15 seconds wall, for about 6.49 seconds combined wall time.
- Independent validation scoring: about 0.016 seconds
- Determinism: repeated 798-choice payloads produced identical SHA-256 hashes
- Reproducibility manifest: 27 transitive local inference-source files and both
  checkpoint assets are hashed in `runs/model_z_robust_semantic/source-manifest.json`
- A live seven-row prediction loaded exactly those 27 project modules and no
  unrecorded local module; see `runtime-dependency-audit.json`.
- Fresh temporary retraining reproduced both asset files byte-for-byte, the
  full selected payload hash and 768/798 validation; see `rebuild-audit.json`.
- One-thread and four-thread selections are identical. Random candidate
  reordering across five fixed seeds changes no selected answer group after
  mapping back; same-group candidate indices may change because the short
  circuit selects the first visible member of the unique group.
- Regression checks: all model/experiment test scripts pass

The memory figure is total Python process RSS, not inference-asset size. The
assignment constrains additional inference assets to 16 MiB and end-to-end CPU
time to 30 seconds; Model Z is comfortably below both measured limits locally.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_z_robust_semantic \
  --split validation \
  --output runs/model_z_robust_semantic/submission-validation-predictions.json

python score_predictions.py --split validation \
  --predictions runs/model_z_robust_semantic/submission-validation-predictions.json \
  --output runs/model_z_robust_semantic/submission-validation-score.json
```

Before final test scoring, freeze the submitted source version, checkpoint
hashes and report. Do not use test references for any further tuning or model
selection.
