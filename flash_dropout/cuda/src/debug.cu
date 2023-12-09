#include <cute/tensor.hpp>
#include <cutlass/util/device_memory.h>

int main(int argc, char *argv[])
{
    cutlass::DeviceAllocation<cute::half_t> x_ptr(24);
    auto x = cute::make_tensor(cute::make_gmem_ptr(x_ptr.get()), cute::make_layout(cute::make_shape(4, 6)));

    return 0;
}
