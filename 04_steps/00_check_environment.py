"""S00: Check the runtime, data location, GPU, and output permissions."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import environment_info, load_config, resolve_path, utc_now, write_json

REQUIRED = ("numpy", "pandas", "sklearn", "torch", "transformers", "yaml", "matplotlib", "seaborn", "joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/01_base.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="07_metadata/environment_check.json")
    args = parser.parse_args()
    config = load_config(args.config)
    imports = {}
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            imports[name] = "ok"
        except ImportError as error:
            imports[name] = f"missing: {error}"
    data_root = resolve_path(args.data_root or config["data"]["root"])
    data_files = {name: (data_root / name).is_file() for name in ("data_full.json", "domains.json")}
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at": utc_now(),
        "imports": imports,
        "data_files": data_files,
        "output_writable": True,
        "environment": environment_info(),
    }
    write_json(output, report)
    print(f"Environment record: {output}")
    if any(value != "ok" for value in imports.values()) or not all(data_files.values()):
        raise SystemExit("Environment check failed; see the JSON record.")
    print("Environment and data-file checks passed. No packages were installed.")


if __name__ == "__main__":
    main()
