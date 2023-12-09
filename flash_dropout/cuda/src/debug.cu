#include <cute/tensor.hpp>
#include <cutlass/util/device_memory.h>

int main(int argc, char *argv[])
{
    cutlass::DeviceAllocation<cute::half_t> x(24);
    auto x = cute::make_tensor<cute::half_t>(cute::make_gmem_ptr(x.get()), cute::make_shape(2, 3, 4));

    return 0;
}
