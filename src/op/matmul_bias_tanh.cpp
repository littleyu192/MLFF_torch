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
        std::vector<double const *> x_ptrs_host(batch_size);
        std::vector<double const *> w_ptrs_host(batch_size);
        std::vector<double const *> bias_ptrs_host(batch_size);
        std::vector<double *> output_ptrs_host(batch_size);

        double const * x_ptr = (double const*) x.data_ptr();
        double const * w_ptr = (double const*) w.data_ptr();
        double const * bias_ptr = (double const*) bias.data_ptr();
        double * output_ptr = (double *) out.data_ptr();

        for (int i = 0; i < batch_size; ++i)
        {
            x_ptrs_host[i] = x_ptr;
            w_ptrs_host[i] = w_ptr;
            bias_ptrs_host[i] = bias_ptr;
            output_ptrs_host[i] = output_ptr;
            x_ptr += m * k;
            output_ptr += m * n;
        }

        double const **x_ptrs_device;
        double const **w_ptrs_device;
        double const **bias_ptrs_device;
        double **output_ptrs_device;

        cudaError_t result = cudaSuccess;

        result = cudaMalloc(&x_ptrs_device, batch_size * sizeof(double*));
        CUDA_CHECK(result);

        result = cudaMemcpy(x_ptrs_device, x_ptrs_host.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);
        CUDA_CHECK(result);

        result = cudaMalloc(&w_ptrs_device, batch_size * sizeof(double*));
        CUDA_CHECK(result);

        result = cudaMemcpy(w_ptrs_device, w_ptrs_host.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);
        CUDA_CHECK(result);


        result = cudaMalloc(&bias_ptrs_device, batch_size * sizeof(double*));
        CUDA_CHECK(result);
        
        result = cudaMemcpy(bias_ptrs_device, bias_ptrs_host.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);
        CUDA_CHECK(result);

        result = cudaMalloc(&output_ptrs_device, batch_size * sizeof(double*));
        CUDA_CHECK(result);

        result = cudaMemcpy(output_ptrs_device, output_ptrs_host.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);
        CUDA_CHECK(result);

        // std::cout << "batch " << batch_size << " M " << m << " n " << n << " k " << k << std::endl;

        cutlass_array_dgemm(m, n, k, 1.0, x_ptrs_device, k, w_ptrs_device, n, output_ptrs_device, n, bias_ptrs_device, 1.0, batch_size);
    }
    else
        printf("data type error!");
    
    return;

}