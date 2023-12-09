import torch
from absl import app, flags
from torch.profiler import ProfilerActivity, profile

from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("MNK", 1024, "Problem size")
flags.DEFINE_integer("block_size", 64, "Dropout block size", short_name="b")
flags.DEFINE_string("output", None, "File to save trace", required=True, short_name="o")
flags.DEFINE_integer("warmup", 10, "Number of warmup iterations")
flags.DEFINE_integer("iterations", 3, "Number of iterations")


def main(_):
    M = FLAGS.MNK
    N = FLAGS.MNK
    K = FLAGS.MNK

    A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn((M, N), device="cuda", dtype=torch.float16)

    def f():
        C = blockwise_dropout_matmul(A, B, (FLAGS.block_size, FLAGS.block_size), 0.5)
        C.backward(grad_output)

    for _ in range(FLAGS.warmup):
        f()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
    ) as prof:
        for _ in range(FLAGS.iterations):
            f()

    prof.export_chrome_trace(FLAGS.output)


if __name__ == "__main__":
    app.run(main)
