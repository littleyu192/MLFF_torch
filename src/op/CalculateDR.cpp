#include "CalculateDR.h"
#include <torch/extension.h>

void torch_launch_calculate_DR(
    torch::Tensor &xyz_scater, int64_t batch_size, int64_t natoms, int64_t neigh_num,
    int64_t ntype, int64_t embedingnet_output_dim, torch::Tensor &DR)
{
    auto dtype = xyz_scater.dtype();
    assert(DR.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_calculate_DR<float>(
            static_cast<float *>(xyz_scater.data_ptr()), batch_size, natoms, neigh_num,
            ntype, embedingnet_output_dim, static_cast<float *>(DR.data_ptr()));
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_DR<double>(
            static_cast<double *>(xyz_scater.data_ptr()), batch_size, natoms, neigh_num,
            ntype, embedingnet_output_dim, static_cast<double *>(DR.data_ptr()));
    }
    else
        printf("data type error!");
}

void torch_launch_calculate_DR_grad(
    const torch::Tensor &xyz_scater, int64_t batch_size, int64_t natoms,
    int64_t neigh_num, int64_t ntype, int64_t embedingnet_output_dim,
    const torch::Tensor &grad_output, torch::Tensor &grad)
{
    auto dtype = xyz_scater.dtype();
    assert(grad.dtype() == dtype);
    assert(grad_output.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_calculate_DR_grad<float>(
            static_cast<float *>(xyz_scater.data_ptr()), batch_size, natoms, neigh_num,
            ntype, embedingnet_output_dim, static_cast<float *>(grad_output.data_ptr()),
            static_cast<float *>(grad.data_ptr()));
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_DR_grad<double>(
            static_cast<double *>(xyz_scater.data_ptr()), batch_size, natoms, neigh_num,
            ntype, embedingnet_output_dim,
            static_cast<double *>(grad_output.data_ptr()),
            static_cast<double *>(grad.data_ptr()));
    }
    else
        printf("data type error!");
}

void torch_launch_calculate_DR_second_grad(
    int64_t batch_size, int64_t natoms, double scale, int64_t embedingnet_output_dim,
    const torch::Tensor &xyz_scater, const torch::Tensor &grad_output,
    const torch::Tensor &grad_second, torch::Tensor &dgrad_xyz_scater,
    torch::Tensor &dgrad_gradoutput)
{
    auto dtype = xyz_scater.dtype();
    assert(dgrad_xyz_scater.dtype() == dtype);
    assert(dgrad_gradoutput.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_calculate_DR_second_grad<float>(
            batch_size, natoms, static_cast<float>(scale), embedingnet_output_dim,
            xyz_scater.data_ptr<float>(), grad_output.data_ptr<float>(),
            grad_second.data_ptr<float>(), dgrad_xyz_scater.data_ptr<float>(),
            dgrad_gradoutput.data_ptr<float>());
    }
    else if (dtype == torch::kFloat64)
    {
        launch_calculate_DR_second_grad<double>(
            batch_size, natoms, static_cast<double>(scale), embedingnet_output_dim,
            xyz_scater.data_ptr<double>(), grad_output.data_ptr<double>(),
            grad_second.data_ptr<double>(), dgrad_xyz_scater.data_ptr<double>(),
            dgrad_gradoutput.data_ptr<double>());
    }
    else
        printf("data type error!");
}