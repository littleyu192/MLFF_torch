#ifndef OP_SRC_INCLUDE_CUTLASSGEMM_H_
#define OP_SRC_INCLUDE_CUTLASSGEMM_H_

#include <iostream>
#include <vector>

#include "Helper.h"

cudaError_t cutlass_array_dgemm(
    int m, int n, int k, double alpha, double const *const *A, int lda,
    double const *const *B, int ldb, double *const *C, int ldc,
    double const *const *bias, double beta, int batch_count);

cudaError_t cutlass_batched_dgemm(
    int m, int n, int k, double alpha, double const *A, int lda, int strideA,
    double const *B, int ldb, int strideB, double const *C, int ldc, int strideC,
    double *D, int ldd, int strideD, double beta, int batch_count, bool tanh);

#endif  // OP_SRC_INCLUDE_CUTLASSGEMM_H_
