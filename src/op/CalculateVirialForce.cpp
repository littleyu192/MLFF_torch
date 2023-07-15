#include <torch/extension.h>
#include "CalculateForce.h"

void torch_launch_calculate_virial_force(
    torch::Tensor &nblist, const torch::Tensor &dE, const torch::Tensor &Rij,
    const torch::Tensor &Ri_d, int64_t batch_size, int64_t natoms, int64_t neigh_num,
    const torch::Tensor &virial_force, const torch::Tensor &atom_virial_force)
{
    auto dtype = dE.dtype();
    assert(Rij.dtype() == dtype);
    assert(Ri_d.dtype() == dtype);
    assert(virial_force.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_virial_force<float>(
            nblist.data_ptr<int>(), dE.data_ptr<float>(), Rij.data_ptr<float>(),
            Ri_d.data_ptr<float>(), batch_size, natoms, neigh_num,
            virial_force.data_ptr<float>(), atom_virial_force.data_ptr<float>());
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_virial_force<double>(
            nblist.data_ptr<int>(), dE.data_ptr<double>(), Rij.data_ptr<double>(),
            Ri_d.data_ptr<double>(), batch_size, natoms, neigh_num,
            virial_force.data_ptr<double>(), atom_virial_force.data_ptr<double>());
    }
    else
        printf("data type error!");
}
