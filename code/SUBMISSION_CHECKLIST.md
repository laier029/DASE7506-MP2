# MP2 Submission Checklist

## Verified development state

- [x] Frozen entry point: `submission.py` (Model Z).
- [x] Matching checkpoint: `runs/model_z_robust_semantic/`.
- [x] Fresh asset rebuild is byte-identical and reproduces the full validation
  selection payload.
- [x] Validation: 768/798 = 96.2406015037594%.
- [x] Five question-level OOF schemes: Model Z improves Model X by
  `[10, 7, 8, 8, 5]` correct questions.
- [x] Four-thread validation prediction: about 1.48 seconds internal and
  1.57 seconds wall time on the development Mac.
- [x] Inference assets: 1,575,004 bytes uncompressed.
- [x] Stress prediction plus independent scoring on all 4,755 train rows:
  about 6.49 seconds combined wall time; no test data was used.
- [x] Full regression suite, compilation and dependency checks pass.
- [x] Arbitrary seven-row input returns aligned native integer choices.
- [x] One-thread and four-thread choices are identical.
- [x] Five fixed candidate-reordering seeds change no selected final-answer
  group after mapping back.
- [x] Inference code does not read references or assume fixed public-bank sizes.
- [x] Absolute-path cold start from outside the project directory reproduces
  768/798, so inference does not depend on the current working directory.
- [x] Transitive inference-source and checkpoint-asset hashes are recorded in
  `runs/model_z_robust_semantic/source-manifest.json`.
- [x] Pre-test freeze audit passes all 128 recorded checks without opening test
  inputs or references.
- [x] Original runner, parser, scorer, licenses and train/validation data match
  the release SHA-256 manifest; test files are not opened by this audit.
- [x] All 27 project modules loaded at runtime are covered by the 27-file source
  manifest; no dynamic local dependency is missing.
- [x] Post-freeze public test prediction and score: 384/467 = 82.22698072805139%.
  Test references remained sealed during all training, feature selection,
  ablation and validation analysis; no further tuning is permitted.

## Still intentionally unverified

- [ ] Reference-machine timing. Local timing is evidence of margin, but the
  assignment's Xeon machine is authoritative.
- [x] Final report PDF layout and 10-page limit: 6-page PDF inspected page by
  page; editable DOCX is also included.
- [ ] Public repository URL, frozen source version/commit and checkpoint URL.
  Prepared locally for `https://github.com/laier029/dase7506-mp2` and tag
  `v1.0-model-z`; mark complete only after the public upload is verified.

## Final freeze sequence

1. Finish the report from `REPORT_OUTLINE.md` and include the complete ablation,
   rejected methods, limitations, resource measurements and AI disclosure.
2. Freeze a source commit, regenerate `source-manifest.json`, and record the
   commit identifier beside the checkpoint hashes in `TRAINING.md`.
3. Re-run validation once from a clean environment using the commands in
   `TRAINING.md`.
4. Only after no further tuning is allowed, run the official test prediction
   and independent scorer exactly once using the frozen source and checkpoint.
   **Completed: 384/467; do not run again for method selection.**
5. Save the official test JSON without editing it. Report its `accuracy` field,
   not the validation result or a hand-calculated score. **Completed and
   recorded in `test-evaluation.json`.**
6. Verify repository and checkpoint links from a logged-out/private browser,
   then complete the website-to-GitHub issue submission flow.

Do not modify the parser, scorer or data files, and do not use the test result
to choose another feature, threshold or checkpoint.
