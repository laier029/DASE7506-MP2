"""Verify the frozen bundle without opening test inputs or references."""

import argparse
import hashlib
import json
from pathlib import Path
import re


CODE = Path(__file__).resolve().parents[1]
CHECKPOINT = CODE / "runs/model_z_robust_semantic"
ASSET_LIMIT = 16 * 1024 * 1024
FIXED_DEVELOPMENT_FILES = (
    "check_data.py",
    "common.py",
    "predict.py",
    "score_predictions.py",
    "data/LICENSE.txt",
    "data/NOTICE.txt",
    "data/manifest.json",
    "data/candidates/gsm8k_train.jsonl",
    "data/candidates/gsm8k_train.npz",
    "data/candidates/gsm8k_validation.jsonl",
    "data/candidates/gsm8k_validation.npz",
    "data/gsm8k_train.jsonl",
    "data/gsm8k_validation.jsonl",
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(value):
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check(condition, message, checks):
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/pretest-freeze-audit.json",
    )
    args = parser.parse_args()
    metadata = _load(CHECKPOINT / "metadata.json")
    score = _load(CHECKPOINT / "submission-validation-score.json")
    direct_score = _load(CHECKPOINT / "validation-score.json")
    submission_predictions = _load(
        CHECKPOINT / "submission-validation-predictions.json"
    )
    direct_predictions = _load(CHECKPOINT / "validation-predictions.json")
    one_thread_predictions = _load(
        CHECKPOINT / "validation-predictions-one-thread.json"
    )
    oof = _load(CHECKPOINT / "oof-audit.json")
    manifest = _load(CHECKPOINT / "source-manifest.json")
    fallback = _load(CHECKPOINT / "fallback-audit.json")
    short_circuit = _load(CHECKPOINT / "short-circuit-audit.json")
    order_invariance = _load(CHECKPOINT / "order-invariance-audit.json")
    gate_marginal = _load(CHECKPOINT / "hard-gate-marginal-audit.json")
    rebuild = _load(CHECKPOINT / "rebuild-audit.json")
    runtime_dependencies = _load(CHECKPOINT / "runtime-dependency-audit.json")
    package_manifest = _load(CODE / "PACKAGE_MANIFEST.json")
    checks = []

    _check(metadata["test_split_used_for_selection"] is False, "test not used", checks)
    _check(score["correct"] == 768 and score["n"] == 798, "validation 768/798", checks)
    _check(direct_score["correct"] == 768 and direct_score["n"] == 798, "direct validation 768/798", checks)
    prediction_sets = (
        submission_predictions,
        direct_predictions,
        one_thread_predictions,
    )
    _check(
        all(
            predictions["protocol"] == metadata.get(
                "protocol", "dase7506-mp2-open-hard-verifier-v5"
            )
            and predictions["split"] == "validation"
            for predictions in prediction_sets
        ),
        "prediction protocol and split fields",
        checks,
    )
    _check(
        all(
            len(predictions["ids"]) == len(predictions["selected"]) == 798
            and len(set(predictions["ids"])) == 798
            for predictions in prediction_sets
        ),
        "prediction lengths and unique IDs",
        checks,
    )
    _check(
        all(
            type(choice) is int and 0 <= choice < 4
            for predictions in prediction_sets
            for choice in predictions["selected"]
        ),
        "prediction choices are native integers in range",
        checks,
    )
    _check(
        submission_predictions["ids"]
        == direct_predictions["ids"]
        == one_thread_predictions["ids"],
        "prediction ID order matches",
        checks,
    )
    _check(
        direct_predictions["selected"] == submission_predictions["selected"],
        "direct and frozen submission choices match",
        checks,
    )
    _check(
        direct_predictions["selected"] == one_thread_predictions["selected"],
        "one-thread and four-thread choices match",
        checks,
    )
    _check(
        _payload_sha256(submission_predictions["selected"])
        == metadata["selected_payload_sha256"],
        "selected payload SHA-256",
        checks,
    )
    _check(
        oof["oof_correct_deltas_model_z_over_x"] == [10, 7, 8, 8, 5],
        "five OOF deltas match frozen record",
        checks,
    )
    _check(
        all(delta > 0 for delta in oof["oof_correct_deltas_model_z_over_x"]),
        "every OOF scheme improves",
        checks,
    )
    _check(
        gate_marginal["all_gates_minus_no_gates"]["oof_correct_deltas"]
        == [21, 20, 18, 20, 19],
        "hard-gate OOF marginal matches frozen record",
        checks,
    )
    _check(
        gate_marginal["all_gates_minus_no_gates"]["validation_correct_delta"] == 7,
        "hard-gate validation marginal is +7",
        checks,
    )
    _check(
        short_circuit["validation"]["short_circuited"] == 443
        and short_circuit["validation"]["short_circuit_correct"] == 443,
        "validation short circuit is 443/443",
        checks,
    )
    _check(
        not fallback["train"]["hard_filter_all_groups_removed"]
        and not fallback["validation"]["hard_filter_all_groups_removed"],
        "natural hard filters never remove every group",
        checks,
    )
    _check(
        order_invariance["seeds"] == [7506, 11, 29, 47, 83]
        and not order_invariance["answer_group_changes_after_mapping"],
        "five candidate-reordering seeds preserve answer groups",
        checks,
    )
    _check(
        rebuild["validation"]["selections_identical_to_frozen_checkpoint"]
        and rebuild["validation"]["selected_payload_sha256"]
        == metadata["selected_payload_sha256"],
        "fresh rebuild reproduces frozen selections",
        checks,
    )
    _check(
        all(
            details["identical_to_frozen_checkpoint"]
            and details["sha256"] == metadata["assets"][name]["sha256"]
            for name, details in rebuild["assets"].items()
        ),
        "fresh rebuild reproduces frozen asset hashes",
        checks,
    )
    _check(
        not runtime_dependencies["missing_runtime_modules"]
        and runtime_dependencies["runtime_project_module_count"]
        == runtime_dependencies["manifest_source_file_count"]
        == manifest["source_file_count"],
        "runtime project modules are covered by the source manifest",
        checks,
    )
    asset_bytes = 0
    for name, expected in metadata["assets"].items():
        path = CHECKPOINT / name
        asset_bytes += path.stat().st_size
        _check(path.stat().st_size == expected["bytes"], f"{name} size", checks)
        _check(_sha256(path) == expected["sha256"], f"{name} SHA-256", checks)
    _check(asset_bytes == metadata["asset_bytes_uncompressed"], "asset byte total", checks)
    _check(asset_bytes <= ASSET_LIMIT, "assets at most 16 MiB", checks)

    for record in manifest["source_files"] + manifest["asset_files"]:
        path = CODE / record["path"]
        _check(path.is_file(), f"manifest file exists: {record['path']}", checks)
        _check(path.stat().st_size == record["size_bytes"], f"manifest size: {record['path']}", checks)
        _check(_sha256(path) == record["sha256"], f"manifest hash: {record['path']}", checks)

    inference_text = "\n".join(
        (CODE / record["path"]).read_text(encoding="utf-8")
        for record in manifest["source_files"]
        if record["path"] != "common.py"
    )
    forbidden_inference_patterns = re.compile(
        r"gsm8k_test|read_jsonl\s*\(|score_choices\s*\(|"
        r"load_inputs\s*\(|load_bank\s*\(|(?<!\d)(?:4755|798|467)(?!\d)"
    )
    _check(
        forbidden_inference_patterns.search(inference_text) is None,
        "inference sources contain no reference loaders, test names, or public-bank counts",
        checks,
    )

    for relative in FIXED_DEVELOPMENT_FILES:
        expected = package_manifest["files"][f"code/{relative}"]
        _check(
            _sha256(CODE / relative) == expected,
            f"original fixed file hash: {relative}",
            checks,
        )

    test_outputs = sorted(
        path.relative_to(CODE).as_posix()
        for path in (CODE / "runs").rglob("*")
        if path.is_file()
        and "test" in path.name.lower()
        and ("prediction" in path.name.lower() or "score" in path.name.lower())
    )
    _check(not test_outputs, "no test prediction or score outputs", checks)
    report = {
        "protocol_note": "pre-test audit; no test inputs or references opened",
        "passed": True,
        "checks_passed": len(checks),
        "asset_bytes": asset_bytes,
        "test_output_files": test_outputs,
        "checks": checks,
    }
    output = CODE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"pre-test freeze audit passed: {len(checks)} checks, "
        f"{asset_bytes} asset bytes"
    )


if __name__ == "__main__":
    main()
