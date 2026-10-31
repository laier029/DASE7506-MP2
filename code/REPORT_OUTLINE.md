# MP2 Report Outline

This outline separates verified results from interpretation. Keep the final
report within the assignment's 10-page limit.

## 1. Research question and hypothesis

**Question:** Can verifier accuracy improve when high-precision negative gates
remove demonstrably bad answer groups before a learned scorer ranks survivors?

**Hypothesis:** Hard rejection should be reserved for near-zero-false-positive
signals with a safe fallback. Weaker semantic and arithmetic evidence should be
used only as small survivor-score adjustments.

## 2. Data and protocol

- Four candidate traces and fixed 1024-dimensional embeddings per question.
- Train references for fitting and question-level OOF analysis; validation for
  development; test kept sealed until the final frozen run.
- Primary metric: official final-numeric-answer selection accuracy.
- Resource constraints: four CPU threads, 30 seconds, 16 MiB inference assets.

## 3. Methods

1. Short-circuit the unique parseable answer group; it covered 443/798
   validation questions with 100% correctness.
2. Baselines: fixed candidate, final-answer majority and learned embedding/text
   candidate scorer.
3. Negative stage: unanimous arithmetic conflicts, fractional/nonpositive
   plausibility, and explicit final-declaration conflicts.
4. Survivor stage: group max-plus-mean scoring with calculation coverage,
   occurrence coverage, target retention and implicit-factor evidence.
5. Recovery stage: omitted-factor switch and consistent boxed-answer recovery,
   both with safe fallbacks.
6. Model Z additions: structure-aware lower-answer priors for remainder or
   difference targets and structure-aware terminal arithmetic-dependency
   coverage for cumulative targets (`+0.50` for 3-1, `+0.40` otherwise).

Explain that the final two signals improved all five OOF schemes when combined
(`[10, 7, 8, 8, 5]` over Model X) and fixed one validation answer group without
a harm. Report OOF fixes and harms, not only net deltas.

State the promotion rule explicitly: select on the primary OOF split, require
non-negative results on four additional balanced five-fold schemes, inspect
fix/harm churn, require no validation regression, and audit fallbacks/resources
before freezing. This is stronger evidence than choosing a rule from one
validation fix.

## 4. Results and ablation

Use the full M-to-Z table in `EXPERIMENTS.md`. Highlight:

- Model M: 749/798 (93.86%).
- Model T: 766/798 (95.99%).
- Model X: 767/798 (96.12%); Model Z: 768/798 (96.24%).
- Frozen Model Z post-freeze test: 384/467 (82.23%), from exactly one
  prediction and one scorer run with no subsequent tuning.
- Model Z resources: 1,575,004 asset bytes and about 1.57 seconds wall time on
  the development Mac.

Explain that validation contains 443 unique-answer fast-path rows, while every
test row has answer disagreement. Validation restricted to disagreement rows is
325/355 = 91.55%, still 9.32 percentage points above test, so the fast path
explains only part of the gap. State that this suggests a harder or shifted
test disagreement distribution and does not justify post-test model changes.

Report both improvements and rejected routes. In particular, show that a rule
can fix one validation example yet be rejected when training precision or OOF
stability is poor.

For the two-stage design claim, report that the four hard gates jointly add
`[21, 20, 18, 20, 19]` OOF correct selections and +7 validation selections
relative to the same scorer/recovery path without hard gates (761/798 without,
768/798 with). Also report the
finite-sample precision lower bounds: observed 100% precision does not imply
equal confidence when a gate flags 279 groups versus only 6.

Break the +7 validation gate contribution down by answer structure: +1 for
`3-1`, +2 for `2-2`, +1 for `2-1-1`, +3 for all-distinct and zero when all
four answers agree. The five OOF audits show no correct-to-wrong gate harm in
any structure.

## 5. Error and uncertainty analysis

- Unanimous answers: 379/379 correct; extra verification is unnecessary there.
- Accuracy declines as disagreement increases: 3-1 96.57%, 2-2 91.11%, 2-1-1
  90.00%, all-distinct 85.00%.
- Remaining errors mainly involve semantic composition: omitted baselines,
  incorrect participant/trip/time counts, percent wording and intermediate
  quantities mistaken for requested totals.
- Manual validation-error taxonomy: 11 multiplicity/time/participant cases,
  9 state-transition/baseline-ledger cases, 8 relational-algebra/comparison
  cases, 1 discrete-remainder case and 1 geometry case. Label this as
  diagnostic interpretation, not an estimated test distribution.
- Score margin is useful but not a calibrated probability; recovery rules can
  deliberately select a negative-margin group.

## 6. Limitations and conclusion

- Rule precision is empirical on this course bank, not a universal guarantee.
- The method checks trace evidence but does not fully parse natural-language
  roles or build a complete symbolic solution.
- Validation confidence intervals describe this sample only and do not predict
  the sealed test result.
- The main conclusion is methodological: separate irreversible rejection from
  reversible scoring, require cross-fold evidence, and retain a fallback.

## 7. Reproducibility and disclosure

Include exact commands, dependency versions, seeds, checkpoint hashes, local
hardware (Apple M5 Pro, 48 GiB RAM, macOS arm64, Python 3.12.10) and the
official score JSON. Disclose substantive AI assistance for code drafting,
experiment orchestration, debugging and report structuring; state that all
reported results were executed and checked locally.
