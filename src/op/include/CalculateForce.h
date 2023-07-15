#ifndef OP_SRC_INCLUDE_CALCULATEFORCE_H_
#define OP_SRC_INCLUDE_CALCULATEFORCE_H_

template <typename DType>
void launch_calculate_force(
    const int *nblist, const DType *dE, const DType *Ri_d, const int batch_size,
    const int natoms, const int neigh_num, DType *force);

template <typename DType>
void launch_calculate_virial_force(
    const int *nblist, const DType *dE, const DType *Rij, const DType *Ri_d,
    const int batch_size, const int natoms, const int neigh_num, DType *virial,
    DType *atom_virial);

#endif  // OP_SRC_INCLUDE_CALCULATEFORCE_H_