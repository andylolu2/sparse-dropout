# Logs/notes

Triton has poor performance for older GPUs like Turing / Volta. (https://github.com/openai/triton/issues/2377)

---

CUTLASS resources:
- Discord: https://github.com/NVIDIA/cutlass/issues/1087
- Useful GitHub issues:
    - https://github.com/NVIDIA/cutlass/issues/1028
    - https://github.com/NVIDIA/cutlass/issues/1242
    - https://github.com/NVIDIA/cutlass/issues/1051
    - https://github.com/NVIDIA/cutlass/issues/1017

---

The Flash Attention repository (https://github.com/Dao-AILab/flash-attention) is a really good example of CUTLASS in practice.

---

The Triton implementation spends most of its on CPU. See the profile screenshot below:

![Screenshot from 2024-01-15 12-39-28](https://github.com/andylolu2/flash-dropout/assets/66584117/9402f4ee-82ca-460a-94c9-557c83e17f79)

The CUDA stream (bottom) is almost entirely empty. 

---

![matmul-performance-torch](https://github.com/andylolu2/flash-dropout/assets/66584117/b6685be1-29f1-445b-8030-67f3539b5e6f)

The above plot is for 0% sparsity. We see that for small problem sizes, the Triton implementation is terrible, most likely due to the CPU overhead. For large problem sizes, the Triton implementation doesn't match Dense most likely due to poor support of Triton for Turing GPUs.

All of the above suggests that I need to use CUDA and C++ to remove all the overhead.

## Forward & backward pass details

Forward pass:
- $Y = X W$
    - Shape: `(M N) = (M K) (N K)`
    - Layout: `Row = Row x Row`
    - Sparse version: `Dense = Sparse x Dense`

Backward pass:
- $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$
    - Shape: `(M K) = (M N) (K N)`
    - Layout: `Row = Row x Col`
    - Sparse version: `Sparse = Dense x Dense`
- $\frac{\partial L}{\partial W}^T = X^T \frac{\partial L}{\partial Y}$
    - Shape: `(K N) = (K M) (N M)`
    - Layout: `Col = Col x Col`
    - Sparse version: `Dense = Sparse x Dense`
