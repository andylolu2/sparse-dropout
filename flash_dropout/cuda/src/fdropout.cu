#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <vector>

#include "matmul.cuh"

// CUDA forward declarations

namespace ct = cute;

template <typename scalar_t>
__global__ void add_one(
    ct::Tensor<ct::ViewEngine<ct::gmem_ptr<scalar_t *>>,
               ct::Layout<ct::Shape<size_t>, ct::Stride<ct::_1>>>
        input,
    ct::Tensor<ct::ViewEngine<ct::gmem_ptr<scalar_t *>>,
               ct::Layout<ct::Shape<size_t>, ct::Stride<ct::_1>>>
        output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < input.size()) {
        output[i] = input[i] + 1;
    }
}

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

template <typename scalar_t>
auto at_to_cute2d(torch::Tensor input) {
    CHECK_INPUT(input);
    AT_ASSERTM(input.dim() == 2, "input must be 2-dimensional");

    auto gmem_ptr = ct::make_gmem_ptr(input.data_ptr<scalar_t>());
    return ct::make_tensor(
        gmem_ptr,
        ct::make_layout(ct::make_shape(input.size(0), input.size(1)),
                        ct::make_stride(input.stride(0), input.stride(1))));
}

std::vector<torch::Tensor> forward(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());

    const int threads = 128;
    const int blocks = (A.size(0) + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda", ([&] {
                                   matmul<scalar_t><<<blocks, threads>>>(
                                       at_to_cute2d<scalar_t>(A),
                                       at_to_cute2d<scalar_t>(B),
                                       at_to_cute2d<scalar_t>(C));
                               }));

    return {C};
}

std::vector<torch::Tensor> backward(torch::Tensor input) {
    CHECK_INPUT(input);

    return {input};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LLTM forward (CUDA)");
    m.def("backward", &backward, "LLTM backward (CUDA)");
}