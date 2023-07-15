#include "CalculateForce.h"
#include <torch/extension.h>

void torch_launch_calculate_force(
    torch::Tensor &nblist, const torch::Tensor &dE, const torch::Tensor &Ri_d,
    int64_t batch_size, int64_t natoms, int64_t neigh_num, const torch::Tensor &force)
{
    auto dtype = dE.dtype();
    assert(Ri_d.dtype() == dtype);
    assert(force.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_force<float>(
            nblist.data_ptr<int>(), dE.data_ptr<float>(), Ri_d.data_ptr<float>(),
            batch_size, natoms, neigh_num, force.data_ptr<float>());
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_force<double>(
            nblist.data_ptr<int>(), dE.data_ptr<double>(), Ri_d.data_ptr<double>(),
            batch_size, natoms, neigh_num, force.data_ptr<double>());
    }
    else
        printf("data type error!");
}
