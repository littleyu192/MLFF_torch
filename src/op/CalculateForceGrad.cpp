#include "CalculateForceGrad.h"
#include <torch/extension.h>

void torch_launch_calculate_force_grad(
    torch::Tensor &nblist, const torch::Tensor &Ri_d, const torch::Tensor &net_grad,
    int64_t batch_size, int64_t natoms, int64_t neigh_num, const torch::Tensor &grad)
{
    auto dtype = Ri_d.dtype();
    assert(net_grad.dtype() == dtype);
    assert(grad.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_force_grad(
            nblist.data_ptr<int>(), Ri_d.data_ptr<float>(), net_grad.data_ptr<float>(),
            batch_size, natoms, neigh_num, grad.data_ptr<float>());
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_force_grad(
            nblist.data_ptr<int>(), Ri_d.data_ptr<double>(),
            net_grad.data_ptr<double>(), batch_size, natoms, neigh_num,
            grad.data_ptr<double>());
    }
    else
        printf("data type error!");
}
