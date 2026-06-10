#!/usr/bin/env python
"""Run all three experimental pipelines in sequence.

Order: growth measurement -> reliability assessment -> feedback loop
"""

import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).resolve().parent
    workdir = scripts_dir.parent

    pipelines = [
        ("growth", str(scripts_dir / "run_growth.py")),
        ("reliability", str(scripts_dir / "run_reliability.py")),
        ("feedback", str(scripts_dir / "run_feedback.py")),
    ]

    failed = []
    for name, script in pipelines:
        print(f"\n{'='*60}")
        print(f"  Running {name} pipeline...")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(workdir),
        )
        if result.returncode != 0:
            print(f"  {name} pipeline FAILED (exit code {result.returncode})")
            failed.append(name)
        else:
            print(f"  {name} pipeline completed successfully.")

    if failed:
        print(f"\nFailed pipelines: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll pipelines completed successfully.")


if __name__ == "__main__":
    main()
