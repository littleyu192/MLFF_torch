#ifndef OP_SRC_INCLUDE_KALMANUPDATE_H_
#define OP_SRC_INCLUDE_KALMANUPDATE_H_

template <typename DType>
void launch_kalman_update(
    const int N, const DType alpha, const DType *A, const DType *K, DType *P);

#endif  // OP_SRC_INCLUDE_KALMANUPDATE_H_
