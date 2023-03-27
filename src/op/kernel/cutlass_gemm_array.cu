#include "cutlass_gemm_array.h"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

cudaError_t cutlass_array_dgemm(
    int m,
    int n,
    int k,
    double alpha,
    double const *const *A,
    int lda,
    double const *const *B,
    int ldb,
    double *const *C,
    int ldc,
    double const *const *bias,
    double beta,
    int batch_count)
{

  using RowMajor = typename cutlass::layout::RowMajor;

  // for tensorcore
  // using ThreadblockShape = typename cutlass::gemm::GemmShape<32, 64, 16>;
  // using WarpShape = typename cutlass::gemm::GemmShape<16, 32, 16>;
  // using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 4>;

  // static int const kAlignmentA = 1;
  // static int const kAlignmentB = 1;
  // static int const kStages = 4;

  using Swizzle = typename cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  using ThreadblockShape = typename cutlass::gemm::GemmShape<8, 64, 8>;
  using WarpShape = typename cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = typename cutlass::gemm::GemmShape<1, 1, 1>;

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  static int const kStages = 2;

  using EpilogueOutputOp = typename cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::Tanh, double, 1, double, double>;

  using Operator = typename cutlass::arch::OpMultiplyAdd;

  // using Gemm = cutlass::gemm::device::GemmArray<
  //     double, RowMajor,
  //     double, RowMajor,
  //     double, RowMajor, double,
  //     cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, Swizzle, kStages, kAlignmentA, kAlignmentB, Operator>;
  
  using Gemm = cutlass::gemm::device::GemmArray<
      double, RowMajor,
      double, RowMajor,
      double, RowMajor, double, cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, Swizzle, kStages, kAlignmentA, kAlignmentB, Operator>;

  using Argument = Gemm::Arguments;

  Argument args({m, n, k},
                A,
                lda,
                B,
                ldb,
                bias,
                0,
                C,
                ldc,
                {alpha, beta},
                batch_count);
  
  size_t workspace_size = Gemm::get_workspace_size(args);

  // std::cout << "Workspace " << workspace_size << std::endl;

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(args);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(args);
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);

  return cudaSuccess;
}