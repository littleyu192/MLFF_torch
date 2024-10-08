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

template<typename DType>
void launch_calculate_DR_second_grad(
    const int batch_size,
    const int natoms,
    const DType scale,
    const int embedingnet_output_dim,
    const DType * xyz_scater,
    const DType * grad_output,
    const DType * grad_second,
    DType * dgrad_xyz_scater,
    DType * dgrad_gradoutput
);