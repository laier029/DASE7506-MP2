"""Confirm that the source manifest covers every loaded project module."""

import json
from pathlib import Path
import sys

import numpy as np


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "models"))

from common import load_inputs
from submission import predict


def main():
    bank, arrays = load_inputs("validation")
    indices = np.asarray([22, 32, 44, 90, 97, 617, 788])
    choices = predict(
        [bank[int(index)] for index in indices],
        {"embeddings": arrays["embeddings"][indices]},
        CODE / "runs/model_z_robust_semantic",
    )
    manifest = json.loads(
        (CODE / "runs/model_z_robust_semantic/source-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = {record["path"] for record in manifest["source_files"]}
    loaded = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(CODE).as_posix()
        except ValueError:
            continue
        if path.suffix == ".py" and not relative.startswith(
            (".venv/", "experiments/", "models/test_")
        ):
            loaded.add(relative)
    missing = sorted(loaded - recorded)
    if missing:
        raise AssertionError(f"runtime project modules missing from manifest: {missing}")
    report = {
        "protocol_note": "validation inputs only; no references or test data read",
        "audit_rows": indices.astype(int).tolist(),
        "choices": choices,
        "runtime_project_module_count": len(loaded),
        "manifest_source_file_count": len(recorded),
        "missing_runtime_modules": missing,
        "runtime_project_modules": sorted(loaded),
    }
    output = CODE / "runs/model_z_robust_semantic/runtime-dependency-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
