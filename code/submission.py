"""Frozen submission interface for the Model Z verifier cascade."""

from pathlib import Path
import sys


MODELS = Path(__file__).resolve().parent / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from robust_semantic_verifier import predict as _predict


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    return _predict(bank, arrays, checkpoint)
