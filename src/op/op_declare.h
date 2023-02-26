#include <torch/extension.h>

void torch_launch_calculate_force(torch::Tensor &nblist,
                                  const torch::Tensor &dE,
                                  const torch::Tensor &Ri_d,
                                  int64_t batch_size,
                                  int64_t natoms,
                                  int64_t neigh_num,
                                  const torch::Tensor &force);

void torch_launch_calculate_force_grad(torch::Tensor &nblist,
                                       const torch::Tensor &Ri_d,
                                       const torch::Tensor &net_grad,
                                       int64_t batch_size,
                                       int64_t natoms,
                                       int64_t neigh_num,
                                       const torch::Tensor &grad);

void torch_launch_calculate_virial_force(torch::Tensor &nblist,
                                         const torch::Tensor &dE,
                                         const torch::Tensor &Rij,
                                         const torch::Tensor &Ri_d,
                                         int64_t batch_size,
                                         int64_t natoms,
                                         int64_t neigh_num,
                                         const torch::Tensor &virial_force,
                                         const torch::Tensor &atom_virial_force);

void torch_launch_calculate_virial_force_grad(torch::Tensor &nblist,
                                              const torch::Tensor &Rij,
                                              const torch::Tensor &Ri_d,
                                              const torch::Tensor &net_grad,
                                              int64_t batch_size,
                                              int64_t natoms,
                                              int64_t neigh_num,
                                              const torch::Tensor &grad);

void torch_launch_calculate_DR(torch::Tensor &xyz_scater,
                               int64_t batch_size,
                               int64_t natoms,
                               int64_t neigh_num,
                               int64_t ntype,
                               int64_t embedingnet_output_dim,
                               torch::Tensor &DR);

void torch_launch_calculate_DR_grad(const torch::Tensor &xyz_scater,
                               int64_t batch_size,
                               int64_t natoms,
                               int64_t neigh_num,
                               int64_t ntype,
                               int64_t embedingnet_output_dim,
                               const torch::Tensor &grad_output,
                               torch::Tensor &grad);