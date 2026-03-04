from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'configs' / 'example_static.yaml'



def main() -> None:
    with tempfile.TemporaryDirectory(prefix='mtl_mlp_smoke_') as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        dummy_dir = tmp_dir / 'example_data'
        config_path = tmp_dir / 'smoke_config.yaml'

        with BASE_CONFIG.open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)

        config['data']['train_files'] = [str(dummy_dir / 'train_a.h5'), str(dummy_dir / 'train_b.h5')]
        config['data']['val_files'] = [str(dummy_dir / 'val.h5')]
        config['data']['test_files'] = [str(dummy_dir / 'test.h5')]
        config['output']['dir'] = str(tmp_dir / 'outputs')
        config['output']['experiment_name'] = 'static_demo_smoke'
        config['training']['epochs'] = 1

        with config_path.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / 'scripts' / 'make_dummy_hdf5.py'),
                '--output_dir',
                str(dummy_dir),
                '--vector_dim',
                '3',
            ]
        )
        subprocess.check_call([sys.executable, str(ROOT / 'train.py'), '--config', str(config_path)])


if __name__ == '__main__':
    main()
