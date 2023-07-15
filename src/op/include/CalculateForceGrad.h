#ifndef OP_SRC_INCLUDE_CALCULATEFORCEGRAD_H_
#define OP_SRC_INCLUDE_CALCULATEFORCEGRAD_H_

template <typename DType>
void launch_calculate_force_grad(
    const int *nblist, const DType *Ri_d, const DType *net_grad, const int batch_size,
    const int natoms, const int neigh_num, DType *grad);

template <typename DType>
void launch_calculate_virial_force_grad(
    const int *nblist, const DType *Rij, const DType *Ri_d, const DType *net_grad,
    const int batch_size, const int natoms, const int neigh_num, DType *grad);

#endif  // OP_SRC_INCLUDE_CALCULATEFORCEGRAD_H_
