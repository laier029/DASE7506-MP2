"""Create a reproducibility manifest for the frozen submission bundle."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_module(module):
    relative = Path(*module.split("."))
    candidates = (
        ROOT / relative.with_suffix(".py"),
        ROOT / "models" / relative.with_suffix(".py"),
    )
    return next((path for path in candidates if path.is_file()), None)


def _direct_local_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return [resolved for module in modules if (resolved := _local_module(module))]


def _dependency_closure(entrypoint):
    pending = [entrypoint]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        pending.extend(_direct_local_imports(path))
    return sorted(visited, key=lambda path: str(path.relative_to(ROOT)))


def _record(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", default="submission.py")
    parser.add_argument(
        "--checkpoint", default="runs/model_z_robust_semantic"
    )
    parser.add_argument(
        "--output",
        default="runs/model_z_robust_semantic/source-manifest.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check current source/assets against --output without rewriting it",
    )
    args = parser.parse_args()

    entrypoint = (ROOT / args.entrypoint).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    output = (ROOT / args.output).resolve()
    source_files = _dependency_closure(entrypoint)
    asset_files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    assets = [
        path
        for path in asset_files
        if path.suffix in {".npz", ".joblib", ".pkl", ".pickle"}
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": str(entrypoint.relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_files": [_record(path) for path in source_files],
        "asset_files": [_record(path) for path in assets],
        "source_file_count": len(source_files),
        "source_bytes": sum(path.stat().st_size for path in source_files),
        "asset_file_count": len(assets),
        "asset_bytes": sum(path.stat().st_size for path in assets),
    }
    if args.verify:
        expected = json.loads(output.read_text(encoding="utf-8"))
        comparable_keys = (
            "schema_version",
            "entrypoint",
            "checkpoint",
            "source_files",
            "asset_files",
            "source_file_count",
            "source_bytes",
            "asset_file_count",
            "asset_bytes",
        )
        mismatches = [
            key for key in comparable_keys if expected.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise SystemExit(
                "manifest verification failed for: " + ", ".join(mismatches)
            )
        print(
            f"manifest verified: {len(source_files)} source files, "
            f"{len(assets)} assets"
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
