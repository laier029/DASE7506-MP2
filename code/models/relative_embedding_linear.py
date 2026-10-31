"""Model D: linear verifier using within-question relative embeddings."""

from linear_verifier import predict_with_linear_model


def predict(bank, arrays, checkpoint):
    return predict_with_linear_model(bank, arrays, checkpoint, relative=True)
