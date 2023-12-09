import torch
from absl import app, flags
from torch.profiler import ProfilerActivity, profile

from eval.mlp.model import BasicNet

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 128, "Batch size", short_name="b")
flags.DEFINE_integer("hidden_dim", 1024, "Hidden dimension", short_name="h")
flags.DEFINE_integer("block_size", 64, "Dropout block size", short_name="blk")
flags.DEFINE_string("output", None, "File to save trace", required=True, short_name="o")
flags.DEFINE_integer("warmup", 10, "Number of warmup iterations")
flags.DEFINE_integer("iterations", 3, "Number of iterations")
flags.DEFINE_string("variant", "blockwise[triton]", "Variant")


def main(_):
    x = torch.randn((FLAGS.batch_size, 784), device="cuda", dtype=torch.float16)

    model = BasicNet(
        x,
        num_layers=3,
        hidden_dim=FLAGS.hidden_dim,
        output_dim=10,
        variant=FLAGS.variant,
        p=0.5,
        block_size=(FLAGS.block_size, FLAGS.block_size),
    ).to(x.device, dtype=x.dtype)

    def f():
        loss = model(x).sum()
        loss.backward()

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
