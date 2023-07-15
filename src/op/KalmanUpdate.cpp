#include "KalmanUpdate.h"
#include <torch/extension.h>

void torch_launch_kalmanUpdate(
    const int64_t N, const double alpha, const torch::Tensor &A, const torch::Tensor &K,
    torch::Tensor &P)
{
    auto dtype = K.dtype();
    assert(A.dtype() == dtype);
    assert(P.dtype() == dtype);

    if (dtype == torch::kFloat32)
    {
        launch_kalman_update<float>(
            N, alpha, A.data_ptr<float>(), K.data_ptr<float>(), P.data_ptr<float>());
    }
    else if (dtype == torch::kFloat64)
    {
        launch_kalman_update<double>(
            N, alpha, A.data_ptr<double>(), K.data_ptr<double>(), P.data_ptr<double>());
    }
    else
        printf("data type error!");
}
