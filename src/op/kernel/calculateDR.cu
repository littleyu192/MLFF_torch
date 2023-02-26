#include <iostream>
#include <ATen/cuda/CUDAContext.h>

#include <calculate_DR.h>

template <>
void launch_calculate_DR(
    float *xyz_scater, // batchsize x natoms x 4 x embedingnet_output_dim
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    float *DR // batchsize x natoms x embedingnet_output_dim x 16
)
{
    cublasStatus_t stat;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float scale = float(1.0) / float(neigh_num * ntype);

    // stat = cublasSscal(handle, batch_size * natoms * 4 * embedingnet_output_dim, &scale, xyz_scater, 1);

    // if (stat != CUBLAS_STATUS_SUCCESS)
    // {
    //     std::cout << "CUBLAS scale failed\n";
    //     return;
    // }
    // float alpha = 1.0, beta = 0.0;

    int batchCount = batch_size * natoms;
    float alpha = scale * scale, beta = 0.0;

    // batchCount x 16 x 4, batchCount x 4 x 25
    stat = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, 16, embedingnet_output_dim, 4, &alpha, xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, DR, embedingnet_output_dim, embedingnet_output_dim * 16, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }

    return;
}

template <>
void launch_calculate_DR(
    double *xyz_scater, // batchsize x natoms x 4 x embedingnet_output_dim
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    double *DR // batchsize x natoms x embedingnet_output_dim x 16
)
{
    cublasStatus_t stat;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    double scale = double(1.0) / double(neigh_num * ntype);

    // stat = cublasDscal(handle, batch_size * natoms * 4 * embedingnet_output_dim, &scale, xyz_scater, 1);

    // if (stat != CUBLAS_STATUS_SUCCESS)
    // {
    //     std::cout << "CUBLAS scale failed\n";
    //     return;
    // }

    // double alpha = 1.0, beta = 0.0;

    int batchCount = batch_size * natoms;

    double alpha = scale * scale, beta = 0.0;

    // batchCount x 16 x 4, batchCount x 4 x 25
    stat = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                     16, embedingnet_output_dim, 4,
                                     &alpha, xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, DR, 16, embedingnet_output_dim * 16, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }

    return;
}


template<>
void launch_calculate_DR_grad(
    const float * xyz_scater,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    const float * grad_output,
    float * grad)
{
    cublasStatus_t stat;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    float scale = float(1.0) / float(neigh_num * ntype);

    float alpha = scale, beta = 1.0;

    int batchCount = batch_size * natoms;

     // batchCount x 4 x 16, batchCount x 25 x 16
    stat = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     embedingnet_output_dim, 4, 16,
                                     &alpha, grad_output, 16, 16 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, grad, embedingnet_output_dim, embedingnet_output_dim * 4, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }

    stat = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                     16, 4, embedingnet_output_dim,
                                     &alpha, grad_output, 16, 16 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, grad, embedingnet_output_dim, embedingnet_output_dim * 4, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }

}

template<>
void launch_calculate_DR_grad(
    const double * xyz_scater,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    const double * grad_output,
    double * grad
)
{
    cublasStatus_t stat;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    double scale = double(1.0) / double(neigh_num * ntype);

    double alpha = scale, beta = 0.0;

    int batchCount = batch_size * natoms;

     // batchCount x 25 x 16   batchCount x 16 x 4
    stat = cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     embedingnet_output_dim, 4, 16,
                                     &alpha, grad_output, 16, 16 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, grad, embedingnet_output_dim, embedingnet_output_dim * 4, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }
    
    beta = 1.0;
     // batchCount x 16 x 25   batchCount x 25 x 4
    stat = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     16, 4, embedingnet_output_dim,
                                     &alpha, grad_output, 16, 16 * embedingnet_output_dim,
                                     xyz_scater, embedingnet_output_dim, 4 * embedingnet_output_dim,
                                     &beta, grad, embedingnet_output_dim, embedingnet_output_dim * 4, batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed\n";
        return;
    }
}