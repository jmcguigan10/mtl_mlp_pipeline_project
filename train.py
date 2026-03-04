from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mtl_mlp.config import load_config
from mtl_mlp.data import build_dataloader, build_datasets
from mtl_mlp.models import MultiTaskMLP
from mtl_mlp.training import build_loss_bundle
from mtl_mlp.training.trainer import Trainer
from mtl_mlp.utils import configure_torch_runtime, set_seed


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')





def load_checkpoint(trainer: Trainer, checkpoint_path: str) -> int:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.loss_bundle.load_state_dict(checkpoint.get('loss_bundle_state_dict', {}))
    trainer.balancer.load_state_dict(checkpoint.get('balancer_state_dict', {}))
    if checkpoint.get('optimizer_state_dict') is not None:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if trainer.balancer_optimizer is not None and checkpoint.get('balancer_optimizer_state_dict') is not None:
        trainer.balancer_optimizer.load_state_dict(checkpoint['balancer_optimizer_state_dict'])
    if trainer.scheduler_bundle.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        trainer.scheduler_bundle.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return int(checkpoint.get('epoch', 0))



def main() -> None:
    parser = argparse.ArgumentParser(description='Train the multi-task MLP pipeline.')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--resume', default=None, help='Optional checkpoint to resume from')
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(
        num_threads=config.training.get('cpu_num_threads'),
        num_interop_threads=config.training.get('cpu_num_interop_threads'),
    )
    set_seed(int(config.get('seed', 42)), deterministic=bool(config.training.get('deterministic', False)))

    datasets = build_datasets(config)
    train_loader = build_dataloader(datasets['train'], config, train=True)
    val_loader = build_dataloader(datasets['val'], config, train=False)
    test_loader = build_dataloader(datasets['test'], config, train=False)

    model = MultiTaskMLP(config)
    loss_bundle = build_loss_bundle(config)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_bundle=loss_bundle,
    )

    if args.resume:
        resumed_epoch = load_checkpoint(trainer, args.resume)
        print(f'Resumed checkpoint from epoch {resumed_epoch}: {Path(args.resume).resolve()}')

    metrics = trainer.fit()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
