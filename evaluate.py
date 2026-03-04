from __future__ import annotations

import argparse
import json

import torch

from mtl_mlp.config import load_config
from mtl_mlp.data import build_dataloader, build_datasets
from mtl_mlp.models import MultiTaskMLP
from mtl_mlp.training import build_loss_bundle
from mtl_mlp.training.trainer import Trainer
from mtl_mlp.utils import configure_torch_runtime, save_json, set_seed


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')





def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained multi-task MLP checkpoint.')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--output', default=None, help='Optional path to save metrics JSON')
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(
        num_threads=config.training.get('cpu_num_threads'),
        num_interop_threads=config.training.get('cpu_num_interop_threads'),
    )
    set_seed(int(config.get('seed', 42)), deterministic=bool(config.training.get('deterministic', False)))

    datasets = build_datasets(config)
    loaders = {
        split: build_dataloader(dataset, config, train=(split == 'train'))
        for split, dataset in datasets.items()
    }

    model = MultiTaskMLP(config)
    loss_bundle = build_loss_bundle(config)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        test_loader=loaders['test'],
        loss_bundle=loss_bundle,
    )
    checkpoint = torch_load_checkpoint(args.checkpoint)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.loss_bundle.load_state_dict(checkpoint.get('loss_bundle_state_dict', {}))
    trainer.balancer.load_state_dict(checkpoint.get('balancer_state_dict', {}))

    metrics = trainer.evaluate(loader=loaders[args.split], split_name=args.split)
    if args.output:
        save_json(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
