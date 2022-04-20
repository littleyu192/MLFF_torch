#ifndef DIY_HPP
#define DIY_HPP

#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;
extern "C"
{
    // use fortran-lapack please use ifort not icpc as linker, and add -mkl to link these subroutines.
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* LDA, int* IPIV, int* INFO);
    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* LDA, int* IPIV, double* WORK, int* LWORK, int* INFO);
    // matmul, C = alpha * A*B + beta * C
    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
}

// call fortran subroutine from cpp, diy fortran subroutine binded C name.
extern "C" void f2c_calc_energy_force(int* /*imodel*/,int* /*n_atom*/, int* /*type_atom*/, double* /*lat*/, double* /*x_frac*/, double* /*e_atom*/, double* /*f_atom*/, double* /*e_tot*/, int* /*iflag_reneighbor*/);
    
namespace diy
{
    // inverse a matrix    
    void dinv(double* A, int n)
    {
        int *ipiv = new int[n];
        int lwork = n*n;
        double *work = new double[lwork];
        int info;
    
        dgetrf_(&n,&n,A,&n,ipiv,&info);
        dgetri_(&n,A,&n,ipiv,work,&lwork,&info);
    
        delete[] ipiv;
        delete[] work;
    }

    // memory operators
    // similar to lammps memory.h
    // 1d malloc and free
    template <typename TYPE> TYPE *create(TYPE *&array, int n)
    {                          
        int nbytes = ((int) sizeof(TYPE)) * n;
        array = (TYPE *) malloc(nbytes);
        return array;
    }
    template <typename TYPE> void destroy(TYPE *&array)
    {
        free(array);
        array = nullptr;                                 
    }                       
    
    // 2d malloc and free
    template <typename TYPE> TYPE **create(TYPE **&array, int n1, int n2)
    {
        // lmp similar malloc function for 2d array
        int nbytes = ((int)sizeof(TYPE)) * n1 * n2;
        TYPE *data = (TYPE *)malloc(nbytes);
        nbytes = ((int)sizeof(TYPE *)) * n1;
        array = (TYPE **)malloc(nbytes);
    
        int n = 0;
        for (int i = 0; i < n1; i++)
        {
          array[i] = &data[n];
          n += n2;
        }
        return array;
    }
    
    template <typename TYPE> void destroy(TYPE **&array)
    {
        // lmp similar free function for 2d array
        if (array == nullptr) return;
        free(array[0]);
        free(array);
        array = nullptr;
    }

}
#endif
