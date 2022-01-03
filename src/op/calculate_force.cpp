#include <torch/extension.h>
#include "calculate_force.h"

void torch_launch_calculate_force(torch::Tensor &nblist,
                       const torch::Tensor &dE,
                       const torch::Tensor &Ri_d,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &force
) 
{
    launch_calculate_force(
        (const int *) nblist.data_ptr(),
        (const double *) dE.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        batch_size, natoms, neigh_num,
        (double *) force.data_ptr()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_force", 
          &torch_launch_calculate_force,
          "calculate force kernel warpper");
}

TORCH_LIBRARY(op, m) {
    m.def("calculate_force", torch_launch_calculate_force);
}
