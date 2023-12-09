#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

template <typename scalar_t>
__global__ void add_one(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = input[index] + 1;
}

// std::vector<torch::Tensor> lltm_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell);

// std::vector<torch::Tensor> lltm_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward(
    torch::Tensor input)
{
    CHECK_INPUT(input);
    auto shape = input.sizes();
    input = input.view({-1});

    const int threads = 128;
    const int blocks = (input.size(0) + threads - 1) / threads;
    auto output = torch::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "add_one_cuda", ([&]
                                                              { add_one<scalar_t><<<blocks, threads>>>(
                                                                    input.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                    output.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()); }));

    output = output.view(shape);

    return {output};
}

std::vector<torch::Tensor> backward(
    torch::Tensor input)
{
    CHECK_INPUT(input);

    return {input};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward, "LLTM forward (CUDA)");
    m.def("backward", &backward, "LLTM backward (CUDA)");
}