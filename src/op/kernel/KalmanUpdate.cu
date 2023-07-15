#include "KalmanUpdate.h"

template <typename DType, int ElementsPerThread, int Threads>
__global__ void kalman_update(
    const int N, const DType alpha, const DType *A, const DType *K, DType *P)
{
    const int blockidx = blockIdx.x;
    const int blockidy = blockIdx.y;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const DType beta = A[0];

    int KIndexX = (blockidx * Threads + tidx) * ElementsPerThread;
    int KIndexY = (blockidy * Threads + tidy) * ElementsPerThread;

    int PIndex = KIndexY * N + KIndexX;

    if (KIndexX >= N || KIndexY >= N)
        return;

    DType res[ElementsPerThread][ElementsPerThread];
    DType KX[ElementsPerThread], KY[ElementsPerThread];

#pragma unroll
    for (int i = 0; i < ElementsPerThread; ++i)
    {
        KX[i] = K[KIndexX + i];
        KY[i] = K[KIndexY + i];

#pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j)
        {
            res[i][j] = alpha * P[PIndex + i * N + j];
        }
    }

#pragma unroll
    for (int i = 0; i < ElementsPerThread; ++i)
    {
#pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j)
        {
            res[i][j] -= beta * KX[j] * KY[i];
            P[PIndex + i * N + j] = res[i][j];
        }
    }
}

// P = alpha * P - A * K * K.T
// P (N * N)
// K (N * 1)
// A (1 * 1)
// alpha is scalar
template <typename DType>
void launch_kalman_update(
    const int N, const DType alpha, const DType *A, const DType *K, DType *P)
{
    static constexpr int Threads = 8;
    static constexpr int ElementsPerThread = 2;

    dim3 thread_grid(Threads, Threads);

    if (N % ElementsPerThread)
    {
        const int nblock = (N + Threads - 1) / (Threads);
        dim3 block_grid(nblock, nblock, 1);
        kalman_update<DType, 1, Threads>
            <<<block_grid, thread_grid>>>(N, alpha, A, K, P);
    }
    else
    {
        const int nblock =
            (N + Threads * ElementsPerThread - 1) / (Threads * ElementsPerThread);
        dim3 block_grid(nblock, nblock, 1);
        kalman_update<DType, ElementsPerThread, Threads>
            <<<block_grid, thread_grid>>>(N, alpha, A, K, P);
    }
}

template void launch_kalman_update(
    const int N, const double alpha, const double *A, const double *K, double *P);

template void launch_kalman_update(
    const int N, const float alpha, const float *A, const float *K, float *P);
