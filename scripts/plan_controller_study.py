"""Write a predeclared controller-study plan; never run a controller or inspect outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

from noether_optics.controller_protocol import load_study, study_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/controller_study_v1.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_study(args.config)
    plan = study_plan(config)
    plan["protocol_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    plan["base_config_sha256"] = config["base_config_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(plan, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Wrote {len(plan['candidates'])} planned candidates; no outcomes generated.")


if __name__ == "__main__":
    main()
