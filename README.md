# SparseDrop

SparseDrop is a simple, structured, and hardware-friendly variant of dropout that can benefit from sparsity on GPUs. 

See [report.pdf](report.pdf) for the final report.

See [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the project's goals and motivations.

See [NOTES.md](docs/NOTES.md) for the development logs and notes.

## Reproduce results

This project is developed and run entirely in Docker with VsCode's Dev Container extension. If you are not using VsCode, you can use the [Dockerfile](./.devcontainer/Dockerfile) to rebuild the same container (not tested). 

The Python dependencies are managed with Poetry.

Bare in mind that this project is developed on a Turing architecture GPU (RTX 2060 Max-Q). Older generation GPUs are not supported. Newer generation GPUs *should* compile but they will not achieve near-peak performance (not tested).

To reproduce the experiment results simply run:
```bash
./experiments/mlp_mnist.sh

./experiments/vit_fashion_mnist.sh

./experiments/vit_cifar10.sh

./experiments/llm_shakespeare.sh
```
You can repeat the experiments by changing the `seed` parameter in each script to a different value and rerun them.

## Raw experiment logs/results

### MLP on MNIST

- W&B link: https://wandb.ai/andylolu2/flash-dropout-mlp
- Run ids: 7-25, 31-49

### ViT on Fashion MNIST

- W&B link: https://wandb.ai/andylolu2/flash-dropout-vit
- Run ids: 47-59

### ViT on CIFAR-10

- W&B link: https://wandb.ai/andylolu2/flash-dropout-vit
- Run ids: 60-71

### GPT on Shakespeare

- W&B link: https://wandb.ai/andylolu2/flash-dropout-llm
- Run ids: 6-19
