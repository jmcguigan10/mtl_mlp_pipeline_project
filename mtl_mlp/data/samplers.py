from __future__ import annotations

import math
import random
from typing import Iterator

from torch.utils.data import Sampler


class ContiguousBlockBatchSampler(Sampler[list[int]]):
    """Yield mini-batches by iterating contiguous index blocks within each file.

    This preserves HDF5 locality for mostly unchunked datasets while still shuffling
    at the block level across files.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        block_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.block_size = int(block_size or max(self.batch_size * 8, 2048))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        if self.batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        if self.block_size < 1:
            raise ValueError('block_size must be >= 1')
        if not hasattr(self.dataset, '_lengths'):
            raise TypeError('ContiguousBlockBatchSampler requires dataset._lengths')

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        for file_len in self.dataset._lengths:
            file_start = start
            file_end = start + int(file_len)
            for block_start in range(file_start, file_end, self.block_size):
                blocks.append((block_start, min(block_start + self.block_size, file_end)))
            start = file_end
        return blocks

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        blocks = self._build_blocks()
        if self.shuffle:
            rng.shuffle(blocks)

        self._epoch += 1
        batch: list[int] = []
        for start, end in blocks:
            for idx in range(start, end):
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)
