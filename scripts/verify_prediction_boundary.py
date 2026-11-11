"""Verify an already downloaded, pinned Prediction Boundary release."""
import argparse
import json
from pathlib import Path

from noether_optics.release_integrity import verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--manifest', type=Path,
                        help='Separate release manifest; default: RUN/release-manifest.json')
    args = parser.parse_args()
    print(json.dumps(verify_release(args.run, manifest_sha256=args.manifest_sha256,
                                    manifest_path=args.manifest), sort_keys=True))


if __name__ == '__main__':
    main()
