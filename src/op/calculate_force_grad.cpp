#include <torch/extension.h>
#include "calculate_force_grad.h"

void torch_launch_calculate_force_grad(torch::Tensor &nblist,
                       const torch::Tensor &Ri_d,
                       const torch::Tensor &net_grad,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &grad
) 
{
    launch_calculate_force_grad(
        (const int *) nblist.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        (const double *) net_grad.data_ptr(),
        batch_size, natoms, neigh_num,
        (double *) grad.data_ptr()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_force_grad", 
          &torch_launch_calculate_force_grad,
          "calculate force grad kernel warpper");
}

TORCH_LIBRARY(op_grad, m) {
    m.def("calculate_force_grad", torch_launch_calculate_force_grad);
}
