import numpy as np
import torch

from structured_dropout.types import size


def dropout_mask(x: torch.Tensor, block_size: size, p: float):
    """Creates a blockwise dropout mask for a matrix.

    mask[i, j] = True means block (i, j) is dropped.
    That is, x[i BLK_N : (i + 1) BLK_N, j BLK_K : (j + 1) BLK_K] is dropped / 0.
    """
    assert x.ndim == 2

    num_blocks = (x.shape[0] // block_size[0], x.shape[1] // block_size[1])
    num_blocks_not_masked = round(num_blocks[1] * (1 - p))

    not_masked_indices = np.argsort(np.random.rand(*num_blocks), axis=1)
    not_masked_indices = np.sort(not_masked_indices[:, :num_blocks_not_masked], axis=1)
    return not_masked_indices


def mask_to_increment_table(mask: np.ndarray, BLOCK_K: int):
    """Converts a mask to an pointer increment table.

    Args:
        mask: A mask of shape (N // BLK_N, K // BLK_K) where True means dropped.
        BLK_K: The block size.

    Example:
        BLK_K = 16
        mask = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]

        table = [
            [0,  16, 16],
            [0,  16, 32],
            [0,  32, 16],
            [16, 16, 16],
        ]
    """
    return np.diff(mask, prepend=0, axis=-1) * BLOCK_K
