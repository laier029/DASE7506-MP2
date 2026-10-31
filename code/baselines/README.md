# Baselines

Both baselines implement the official `predict(bank, arrays, checkpoint)` interface and require no checkpoint.

- `fixed_zero.py`: always selects candidate 0.
- `majority_vote.py`: parses each candidate's final numeric answer and selects the earliest candidate in the largest answer group. Ties are resolved by candidate order; candidate 0 is used if no answer can be parsed.

Run validation from `code/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method baselines/fixed_zero.py --split validation \
  --output runs/baseline_a/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/baseline_a/validation-predictions.json \
  --output runs/baseline_a/validation-score.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method baselines/majority_vote.py --split validation \
  --output runs/baseline_b/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/baseline_b/validation-predictions.json \
  --output runs/baseline_b/validation-score.json
```
