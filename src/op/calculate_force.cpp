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
    auto dtype = dE.dtype();
    assert(Ri_d.dtype() == dtype);
    assert(force.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_force<float>(
            (const int *) nblist.data_ptr(),
            (const float *) dE.data_ptr(),
            (const float *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (float *) force.data_ptr()
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_force<double>(
            (const int *) nblist.data_ptr(),
            (const double *) dE.data_ptr(),
            (const double *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (double *) force.data_ptr()
        );
    }
    else
        printf("data type error!");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_force", 
          &torch_launch_calculate_force,
          "calculate force kernel warpper");
}

TORCH_LIBRARY(op, m) {
    m.def("calculate_force", torch_launch_calculate_force);
}
