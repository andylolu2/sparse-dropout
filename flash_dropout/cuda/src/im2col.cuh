#include <torch/extension.h>

#include <cute/tensor.hpp>

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;
using LayoutNHWC = ct::Layout<ct::Shape<int64_t, int64_t, int64_t, int64_t>, ct::Stride<int64_t, int64_t, int64_t, Int<1>>>;
using LayoutNPQRSC = ct::Layout<ct::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, ct::Stride<int64_t, int64_t, int64_t, int64_t, int64_t, Int<1>>>;

template <typename T>
// Each threadblock is responsible for channel of the input tensor
// i.e. parrallelised over N, H, W
// Each thread is responsible for a 128-bit chunk of the input tensor
__global__ void im2col_kernel(
    ct::Tensor<Gmem<T>, LayoutNHWC> input,
    ct::Tensor<Gmem<T>, LayoutNPQRSC> output,
    int64_t R, int64_t S,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w) {
    static constexpr int64_t ElemsPerThread = 128 / ct::sizeof_bits_v<T>;

    int64_t N = ct::size<0>(input);
    int64_t H = ct::size<1>(input);
    int64_t W = ct::size<2>(input);
    int64_t C = ct::size<3>(input);
    int64_t P = ct::size<1>(output);
    int64_t Q = ct::size<2>(output);

    // Global threadblock shape
    auto threadblock_shape = ct::make_shape(N, H, W);
    // Current threadblock coordinates
    auto nhw = ct::idx2crd(blockIdx.x, threadblock_shape);
    auto n = ct::get<0>(nhwc);
    auto h = ct::get<1>(nhwc);
    auto w = ct::get<2>(nhwc);

    // Load input from gmem -> rmem
    auto r_input = ct::make_tensor<T>(Int<ElemsPerThread>{});
    auto g_input = ct::local_tile(input(n, h, w), ct::Shape<Int<ElemsPerThread>>{}, threadIdx.x);
    ct::copy(ct::SM80_CP_ASYNC_CACHEALWAYS<ct::uint128_t>{}, g_input, r_input);
    ct::cp_async_wait<0>();

    // Main loop
    for (int64_t r = 0; r < R; r++) {
        for (int64_t s = 0; s < S; s++) {
            // Calculate the output coordinate that uses input(n, h, w) with kernel(r, s)
            // If stride is not 1, there might not be one
            // h = p * stride_h + r * dilation_h - padding_h
            // => p = (h - r * dilation_h + padding_h) / stride_h
            // w = q * stride_w + s * dilation_w - padding_w
            // => q = (w - s * dilation_w + padding_w) / stride_w
            int64_t p_unstrided = h - r * dilation_h + padding_h;
            int64_t q_unstrided = w - s * dilation_w + padding_w;
            int64_t p = p_unstrided / stride_h;
            int64_t q = q_unstrided / stride_w;
            if ((p_unstrided % stride_h) != 0 || (q_unstrided % stride_w) != 0 || p < 0 || p >= P || q < 0 || q >= Q) {
                continue;
            }
            auto g_output = ct::local_tile(output(n, p, q, r, s, _), ct::Shape<Int<ElemsPerThread>>{}, threadIdx.x);
            ct::copy(ct::SM80_CP_ASYNC_CACHEALWAYS<ct::uint128_t>{}, r_input, g_output);
        }
    }
    ct::cp_async_wait<0>();
}

template <typename T>
void im2col(
    ct::Tensor<Gmem<T>, LayoutNHWC> input,
    ct::Tensor<Gmem<T>, LayoutNPQRSC> output,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w) {
    int64_t N = ct::size<0>(input);
    int64_t H = ct::size<1>(input);
    int64_t W = ct::size<2>(input);
    int64_t C = ct::size<3>(input);
    assert(128 % ct::sizeof_bits_v<T> == 0);
    assert(C % (128 / ct::sizeof_bits_v<T>) == 0);

    dim3 block_dim(N * H * W);
    dim3 thread_dim(C / (128 / ct::sizeof_bits_v<T>));
    im2col_kernel<<<block_dim, thread_dim>>>(input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
}

void im2col_torch(
    torch::Tensor input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t P = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t Q = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    TORCH_CHECK(input.stride(1) == 1, "Input must be channel-last");

    auto output = torch::empty({N, P, Q, kernel_h, kernel_w, C}, input.options());

    AT_DISPATCH_ALL_TYPES(input.type(), "im2col", [&] {
        auto input_ct = ct::make_tensor(
            ct::make_gmem_ptr(reinterpret_cast<scalar_t *>(input.data_ptr())),
            ct::make_shape(N, H, W, C),
            ct::make_stride(input.stride(0), input.stride(2), input.stride(3), Int<1>{}));
        auto output_ct = ct::make_tensor(
            ct::make_gmem_ptr(reinterpret_cast<scalar_t *>(output.data_ptr())),
            ct::make_shape(N, P, Q, kernel_h, kernel_w, C),
            ct::make_stride(output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4), Int<1>{}));
        im2col(input_ct, output_ct, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("im2col", &im2col_torch, "im2col");
}