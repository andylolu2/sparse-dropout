import torch

torch.set_printoptions(sci_mode=False, precision=4, linewidth=10000, edgeitems=5)

M, N, K = 64, 32, 32

A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) / 100
B = -torch.arange(K * N, dtype=torch.float32).reshape(N, K) / 100

A = A.to(device="cuda", dtype=torch.float16)
B = B.to(device="cuda", dtype=torch.float16)
C = A @ B.T

# print(A)
# print(B)
print(C)
