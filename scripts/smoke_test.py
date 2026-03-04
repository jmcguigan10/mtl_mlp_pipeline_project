from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DUMMY_DIR = ROOT / 'example_data'
CONFIG = ROOT / 'configs' / 'example_static.yaml'



def main() -> None:
    if DUMMY_DIR.exists():
        shutil.rmtree(DUMMY_DIR)
    subprocess.check_call([sys.executable, str(ROOT / 'scripts' / 'make_dummy_hdf5.py'), '--output_dir', str(DUMMY_DIR), '--vector_dim', '3'])
    subprocess.check_call([sys.executable, str(ROOT / 'train.py'), '--config', str(CONFIG)])


if __name__ == '__main__':
    main()
