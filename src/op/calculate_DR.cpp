#include <torch/extension.h>
#include "calculate_DR.h"

void torch_launch_calculate_DR(torch::Tensor &xyz_scater,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       int64_t ntype,
                       int64_t embedingnet_output_dim,
                       torch::Tensor &DR
) 
{
    auto dtype = xyz_scater.dtype();
    assert(DR.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_calculate_DR<float>(
            (float *) xyz_scater.data_ptr(),
            batch_size, natoms, neigh_num, ntype, embedingnet_output_dim,
            (float *) DR.data_ptr()
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_DR<double>(
            (double *) xyz_scater.data_ptr(),
            batch_size, natoms, neigh_num, ntype, embedingnet_output_dim,
            (double *) DR.data_ptr()
        );
    }
    else
        printf("data type error!");
}

void torch_launch_calculate_DR_grad(const torch::Tensor &xyz_scater,
                               int64_t batch_size,
                               int64_t natoms,
                               int64_t neigh_num,
                               int64_t ntype,
                               int64_t embedingnet_output_dim,
                               const torch::Tensor &grad_output,
                               torch::Tensor &grad)
{
    auto dtype = xyz_scater.dtype();
    assert(grad.dtype() == dtype);
    assert(grad_output.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_calculate_DR_grad<float>(
            (float *) xyz_scater.data_ptr(),
            batch_size, natoms, neigh_num, ntype, embedingnet_output_dim,
            (float *) grad_output.data_ptr(), (float *) grad.data_ptr()
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_DR_grad<double>(
            (double *) xyz_scater.data_ptr(),
            batch_size, natoms, neigh_num, ntype, embedingnet_output_dim,
            (double *) grad_output.data_ptr(), (double *) grad.data_ptr()
        );
    }
    else
        printf("data type error!");
}
