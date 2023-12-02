# Structured Dropout project plan

## Motivation

Dropout is a common operation in machine learning to prevent overfitting. It does so by randomly dropping activations of a given layer. The standard implementation does so by setting random activations to zero, which is results in a sparse activation matrix. Unfortunately, such activations are treated as dense matrix for downstream operations which will perform redundant computation. 

In this project, we will implement structured dropout, which will drop activations in a structured way. This will allow us to perform efficient sparse matrix multiplication and other operations on the sparse activations, which will result in higher throughput.

## Related work

Structured dropout (e.g. DropBlock / DropChannel) is not a new concept. However, prior works are motivated by accuracy where standard dropout is not sufficient (especially for CNNs) and neglect the potential speedup due to sparsity. This project will realise this speedup by fusing sparse matrix multiplication with structured dropout.

## Goals

Core:

- Implement structured dropout + sparse matrix multiplication in Triton (or CUDA if necessary). (Both forward and backward pass)
- Verify that the implementation's correctness and benchmark the promised speedup.
- Test the implementation on small/medium-sized models (e.g. GPT-2 small) and compare the speed and final performance with the standard dropout (baseline).

Extensions:
- Implement structured dropout + sparse convolution and benchmark the speedup.
- Test the implementation on vision models (e.g. ResNet-50).

## Potential models

- T5: Uses dropout in the middle of the MLP layers.
- Many models apply dropout after embedding layer.
- Can focus on the pre-QKV and pre-MLP dropout layers.

## Dense forward & backward pass

$$
\begin{align*}
Y &= X W \\
\frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} W^T \\
\frac{\partial L}{\partial W} &= X^T \frac{\partial L}{\partial Y} \\
\end{align*}
$$
