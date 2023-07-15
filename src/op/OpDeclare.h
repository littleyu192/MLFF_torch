#ifndef OP_SRC_OPDECLARE_H_
#define OP_SRC_OPDECLARE_H_

#include <torch/extension.h>

void torch_launch_calculate_force(
    torch::Tensor &nblist, const torch::Tensor &dE, const torch::Tensor &Ri_d,
    int64_t batch_size, int64_t natoms, int64_t neigh_num, const torch::Tensor &force);

void torch_launch_calculate_force_grad(
    torch::Tensor &nblist, const torch::Tensor &Ri_d, const torch::Tensor &net_grad,
    int64_t batch_size, int64_t natoms, int64_t neigh_num, const torch::Tensor &grad);

void torch_launch_calculate_virial_force(
    torch::Tensor &nblist, const torch::Tensor &dE, const torch::Tensor &Rij,
    const torch::Tensor &Ri_d, int64_t batch_size, int64_t natoms, int64_t neigh_num,
    const torch::Tensor &virial_force, const torch::Tensor &atom_virial_force);

void torch_launch_calculate_virial_force_grad(
    torch::Tensor &nblist, const torch::Tensor &Rij, const torch::Tensor &Ri_d,
    const torch::Tensor &net_grad, int64_t batch_size, int64_t natoms,
    int64_t neigh_num, const torch::Tensor &grad);

void torch_launch_calculate_DR(
    torch::Tensor &xyz_scater, int64_t batch_size, int64_t natoms, int64_t neigh_num,
    int64_t ntype, int64_t embedingnet_output_dim, torch::Tensor &DR);

void torch_launch_calculate_DR_grad(
    const torch::Tensor &xyz_scater, int64_t batch_size, int64_t natoms,
    int64_t neigh_num, int64_t ntype, int64_t embedingnet_output_dim,
    const torch::Tensor &grad_output, torch::Tensor &grad);

void torch_launch_calculate_DR_second_grad(
    int64_t batch_size, int64_t natoms, double scale, int64_t embedingnet_output_dim,
    const torch::Tensor &xyz_scater, const torch::Tensor &grad_output,
    const torch::Tensor &grad_second, torch::Tensor &dgrad_xyz_scater,
    torch::Tensor &dgrad_gradoutput);

void torch_launch_kalmanUpdate(
    const int64_t N, const double alpha, const torch::Tensor &A, const torch::Tensor &K,
    torch::Tensor &P);

void torch_launch_matmul_bias_tanh(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w,
    const torch::Tensor &bias, torch::Tensor &out);

void torch_launch_matmul_bias(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w,
    const torch::Tensor &bias, torch::Tensor &out);

void torch_launch_matmul(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w, bool transX,
    bool transW, bool broadcastX, bool broadcastW, torch::Tensor &out);

#endif  // OP_SRC_OPDECLARE_H_
