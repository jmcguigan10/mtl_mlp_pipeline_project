from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mtl_mlp.config import load_config
from mtl_mlp.data import MultiFileHDF5Dataset, build_dataloader, build_datasets
from mtl_mlp.models import MultiTaskMLP
from mtl_mlp.utils import configure_torch_runtime, move_batch_to_device, set_seed


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')





def _build_loader_from_args(config, split: str, files: list[str] | None):
    if files:
        dataset = MultiFileHDF5Dataset(
            files=files,
            key_map={
                'input': config.data.keys.input,
                'bc_target': config.data.keys.bc_target,
                'vector_target': config.data.keys.vector_target,
                'reg_target': config.data.keys.reg_target,
                'sample_weight': config.data.keys.get('sample_weight'),
            },
            strict=bool(config.data.get_path('hdf5.strict', True)),
            swmr=bool(config.data.get_path('hdf5.swmr', False)),
            require_targets=False,
        )
        return build_dataloader(dataset, config, train=False)

    datasets = build_datasets(config)
    return build_dataloader(datasets[split], config, train=False)



def main() -> None:
    parser = argparse.ArgumentParser(description='Run inference and save predictions to NPZ.')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Configured split to run')
    parser.add_argument('--files', nargs='*', default=None, help='Optional explicit HDF5 files to run instead of a config split')
    parser.add_argument('--output', required=True, help='Output .npz path')
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(
        num_threads=config.training.get('cpu_num_threads'),
        num_interop_threads=config.training.get('cpu_num_interop_threads'),
    )
    set_seed(int(config.get('seed', 42)), deterministic=bool(config.training.get('deterministic', False)))

    loader = _build_loader_from_args(config, args.split, args.files)
    if loader is None:
        raise ValueError('No dataset available for prediction.')

    model = MultiTaskMLP(config)
    checkpoint = torch_load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() and str(config.training.get('device', 'auto')).lower() != 'cpu' else 'cpu')
    model.to(device)
    model.eval()

    bc_logits = []
    bc_probs = []
    vector_preds = []
    reg_preds = []
    file_indices = []
    sample_indices = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(batch['inputs'])
        bc_logit = outputs['bc'].detach().cpu().numpy()
        bc_logits.append(bc_logit)
        if bc_logit.shape[-1] == 1:
            bc_probs.append(torch.sigmoid(outputs['bc']).detach().cpu().numpy())
        else:
            bc_probs.append(torch.softmax(outputs['bc'], dim=-1).detach().cpu().numpy())
        vector_preds.append(outputs['vector_regression'].detach().cpu().numpy())
        reg_preds.append(outputs['regression'].detach().cpu().numpy())
        file_indices.append(raw_batch['file_index'].cpu().numpy())
        sample_indices.append(raw_batch['sample_index'].cpu().numpy())

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        bc_logits=np.concatenate(bc_logits, axis=0),
        bc_probs=np.concatenate(bc_probs, axis=0),
        vector_regression=np.concatenate(vector_preds, axis=0),
        regression=np.concatenate(reg_preds, axis=0),
        file_index=np.concatenate(file_indices, axis=0),
        sample_index=np.concatenate(sample_indices, axis=0),
    )
    print(f'Saved predictions to {output_path}')


if __name__ == '__main__':
    main()
