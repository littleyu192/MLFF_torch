#include <torch/extension.h>

#include "CalculateDR.h"
#include "CutlassGemm.h"

void torch_launch_matmul_bias_tanh(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w,
    const torch::Tensor &bias, torch::Tensor &out)
{
    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = x.size(-2);
    const int k = x.size(-1);
    const int n = w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = x.data_ptr<double>();
        double const *w_ptr = w.data_ptr<double>();
        double const *bias_ptr = bias.data_ptr<double>();
        double *output_ptr = out.data_ptr<double>();

        cutlass_batched_dgemm(
            m, n, k, 1.0, x_ptr, k, m * k, w_ptr, n, 0, bias_ptr, 0, 0, output_ptr, n,
            m * n, 1.0, batch_size, true);
    }
    else
        printf("data type error!");

    return;
}

void torch_launch_matmul_bias(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w,
    const torch::Tensor &bias, torch::Tensor &out)
{
    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = x.size(-2);
    const int k = x.size(-1);
    const int n = w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = x.data_ptr<double>();
        double const *w_ptr = w.data_ptr<double>();
        double const *bias_ptr = bias.data_ptr<double>();
        double *output_ptr = out.data_ptr<double>();

        cutlass_batched_dgemm(
            m, n, k, 1.0, x_ptr, k, m * k, w_ptr, n, 0, bias_ptr, 0, 0, output_ptr, n,
            m * n, 1.0, batch_size, false);
    }
    else
        printf("data type error!");

    return;
}

void torch_launch_matmul(
    int64_t batch_size, const torch::Tensor &x, const torch::Tensor &w, bool transX,
    bool transW, bool broadcastX, bool broadcastW, torch::Tensor &out)
{
    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = transX ? x.size(-1) : x.size(-2);
    const int k = transX ? x.size(-2) : x.size(-1);
    const int n = transW ? w.size(-2) : w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = x.data_ptr<double>();
        double const *w_ptr = w.data_ptr<double>();
        double *output_ptr = out.data_ptr<double>();

        cublasStatus_t stat;

        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        const double alpha = 1.0, beta = 0.0;

        stat = cublasDgemmStrideBatchedRowMajor(
            handle, transX ? CUBLAS_OP_T : CUBLAS_OP_N,
            transW ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, x_ptr, x.size(-1),
            broadcastX ? 0 : m * k, w_ptr, w.size(-1), broadcastW ? 0 : k * n, &beta,
            output_ptr, n, m * n, batch_size);

        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "CUBLAS gemm failed\n";
            return;
        }
    }
    else
        printf("data type error!");

    return;
}
