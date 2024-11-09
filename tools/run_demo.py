"""Run the full IARS pipeline and print a summary.

    python tools/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iars.pipeline import run_all  # noqa: E402


def main() -> None:
    result = run_all(seed=7)
    summary = result.summary()
    print("Swift IARS - pipeline summary")
    print("=" * 46)
    print(json.dumps(summary, indent=2, default=lambda o: float(round(o, 4))))
    print()
    print("Navigation updates accepted by the ESKF:")
    for k, v in summary["navigation"]["updates"].items():
        print(f"  {k:>10}: {v}")


if __name__ == "__main__":
    main()
