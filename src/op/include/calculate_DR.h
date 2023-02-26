template<typename DType>
void launch_calculate_DR(
    DType * xyz_scater,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    DType * DR
);

template<typename DType>
void launch_calculate_DR_grad(
    const DType * xyz_scater,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int ntype,
    const int embedingnet_output_dim,
    const DType * grad_output,
    DType * grad
);