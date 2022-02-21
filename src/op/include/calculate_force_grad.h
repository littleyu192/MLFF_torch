void launch_calculate_force_grad(
    const int * nblist,
    const double * Ri_d,
    const double * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * grad
);