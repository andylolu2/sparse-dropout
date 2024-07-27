import torch


def make_tensor(M: int, N: int, row_major: bool = True, requires_grad: bool = True):
    if row_major:
        return torch.randn(
            (M, N), device="cuda", dtype=torch.float16, requires_grad=requires_grad
        )
    else:
        return torch.randn(
            (N, M), device="cuda", dtype=torch.float16, requires_grad=requires_grad
        ).T
