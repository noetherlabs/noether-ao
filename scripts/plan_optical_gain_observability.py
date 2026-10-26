"""Write the frozen P2 plan without simulating, calibrating or reading outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from noether_optics.observability_protocol import load_protocol, protocol_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/optical_gain_observability_v1.yaml")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_protocol(args.config)
    plan = protocol_plan(config)
    plan["protocol_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    plan["base_config_sha256"] = config["base_config_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(plan, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        f"Wrote {plan['maximum_planned_trials']} planned trials; "
        "no threshold or outcome generated."
    )


if __name__ == "__main__":
    main()
