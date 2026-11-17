"""Compare two metrics files. Exit 1 if any number differs by more than a relative 1e-4.

    python scripts/compare_metrics.py committed.json results/metrics.json

Used in CI: the study is rerun on a fresh machine and must reproduce the committed results.
The tolerance allows for last digit differences between processors, nothing more.
"""
import json
import sys

TOL = 1e-4


def walk(a, b, path, bad):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: missing on one side")
            else:
                walk(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > TOL * max(abs(a), abs(b), 1e-3):
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")


def main() -> None:
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    bad = []
    walk(a, b, "", bad)
    if bad:
        print("\n".join(bad[:40]))
        sys.exit(1)
    print("results reproduce")


if __name__ == "__main__":
    main()
