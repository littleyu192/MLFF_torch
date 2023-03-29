#include <torch/extension.h>
#include "cutlass_gemm_array.h"

void torch_launch_matmul_bias_tanh(int64_t batch_size,
                                   const torch::Tensor &x,
                                   const torch::Tensor &w,
                                   const torch::Tensor &bias,
                                   torch::Tensor &out)
{

    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = x.size(-2);
    const int k = x.size(-1);
    const int n = w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = (double const *)x.data_ptr();
        double const *w_ptr = (double const *)w.data_ptr();
        double const *bias_ptr = (double const *)bias.data_ptr();
        double *output_ptr = (double *)out.data_ptr();

        cutlass_batched_dgemm(m, n, k, 1.0, x_ptr, k, m * k, w_ptr, n, 0, bias_ptr, 0, 0, output_ptr, n, m * n, 1.0, batch_size, true);
    }
    else
        printf("data type error!");

    return;
}

void torch_launch_matmul_bias(int64_t batch_size,
                              const torch::Tensor &x,
                              const torch::Tensor &w,
                              const torch::Tensor &bias,
                              torch::Tensor &out)
{

    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = x.size(-2);
    const int k = x.size(-1);
    const int n = w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = (double const *)x.data_ptr();
        double const *w_ptr = (double const *)w.data_ptr();
        double const *bias_ptr = (double const *)bias.data_ptr();
        double *output_ptr = (double *)out.data_ptr();

        cutlass_batched_dgemm(m, n, k, 1.0, x_ptr, k, m * k, w_ptr, n, 0, bias_ptr, 0, 0, output_ptr, n, m * n, 1.0, batch_size, false);
    }
    else
        printf("data type error!");

    return;
}

void torch_launch_matmul(int64_t batch_size,
                         const torch::Tensor &x,
                         const torch::Tensor &w,
                         torch::Tensor &out)
{

    auto dtype = w.dtype();
    assert(x.dtype() == dtype);
    assert(out.dtype() == dtype);

    const int m = x.size(-2);
    const int k = x.size(-1);
    const int n = w.size(-1);

    if (dtype == torch::kFloat64)
    {
        double const *x_ptr = (double const *)x.data_ptr();
        double const *w_ptr = (double const *)w.data_ptr();
        double *output_ptr = (double *)out.data_ptr();

        cutlass_batched_dgemm(m, n, k, 1.0, x_ptr, k, m * k, w_ptr, n, 0, output_ptr, n, m * n, output_ptr, n, m * n, 1.0, batch_size, false);
    }
    else
        printf("data type error!");

    return;
}