"""Run the full analysis end to end.

    python run_pipeline.py                 # everything, including downloads
    python run_pipeline.py --skip-download # reuse data already in data/raw/
    python run_pipeline.py --notebooks     # also re-execute the notebooks
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

DOWNLOAD_STEPS = [
    ["get_chembl_data.py", "--all"],
    ["target_evidence.py"],
]
ANALYSIS_STEPS = [
    ["process_activities.py", "--all"],
    ["selectivity.py"],
    ["model.py"],
    ["prioritise.py"],
]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--notebooks", action="store_true")
    args = parser.parse_args()

    steps = ([] if args.skip_download else DOWNLOAD_STEPS) + ANALYSIS_STEPS
    for script, *script_args in steps:
        run([sys.executable, str(SRC / script), *script_args])

    if args.notebooks:
        for nb in sorted((ROOT / "notebooks").glob("0[2-9]_*.ipynb")):
            run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--inplace", nb.name], cwd=nb.parent)


if __name__ == "__main__":
    main()
