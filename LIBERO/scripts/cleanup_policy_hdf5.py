"""Delete all rollout HDF5 directories under Policy_output.

Client-side per-rollout HDF5s (Policy_output/.../HDF5/) have no downstream
consumer — VLM eval reads training HDF5s from dataset/, not Policy_output/.
Safe to wipe.

Default is dry-run. Pass --yes to actually delete.

Usage:
    python LIBERO/scripts/cleanup_policy_hdf5.py              # dry-run, just list
    python LIBERO/scripts/cleanup_policy_hdf5.py --yes        # actually delete
    python LIBERO/scripts/cleanup_policy_hdf5.py --root /alt  # override root
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "Policy_output")


def _dir_size(path: str) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def find_hdf5_dirs(root: str) -> list[str]:
    """Match exactly directories named 'HDF5' at any depth."""
    matches: list[str] = []
    for base, dirs, _ in os.walk(root):
        # Don't descend into an HDF5 dir once matched (avoid double-counting)
        if "HDF5" in dirs:
            matches.append(os.path.join(base, "HDF5"))
            dirs.remove("HDF5")
    return matches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"Root dir to scan (default: {DEFAULT_ROOT})")
    ap.add_argument("--yes", action="store_true", help="Actually delete. Without this, dry-run.")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: {args.root} is not a directory.", file=sys.stderr)
        return 2

    dirs = find_hdf5_dirs(args.root)
    if not dirs:
        print(f"No HDF5/ directories under {args.root}.")
        return 0

    total = 0
    print(f"{'DELETE' if args.yes else 'DRY-RUN — would delete'}  ({len(dirs)} directories under {args.root})")
    print()
    for d in dirs:
        size = _dir_size(d)
        total += size
        print(f"  {size/1e6:>9.1f} MB   {d}")

    print()
    print(f"Total: {total/1e9:.2f} GB across {len(dirs)} HDF5/ directories")

    if not args.yes:
        print("\nPass --yes to actually delete.")
        return 0

    # Really delete
    n_ok = 0
    for d in dirs:
        try:
            shutil.rmtree(d)
            n_ok += 1
        except Exception as e:
            print(f"  failed to delete {d}: {e}", file=sys.stderr)
    print(f"\nDeleted {n_ok}/{len(dirs)} directories, freed ~{total/1e9:.2f} GB")
    return 0 if n_ok == len(dirs) else 1


if __name__ == "__main__":
    sys.exit(main())
