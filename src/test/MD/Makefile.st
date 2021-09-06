.SUFFIXES: .f .f90 .cu
.f.o:
  $(FC) $(FFLAGS) -c $<

.f90.o:
  $(FC) $(F90FLAGS) -c $<

.cu.o:
  $(NVCC) -std=c++11 $(NVCCFLAG) -c -O3 $<

.c.o:
  $(MPICC) $(MPICCFLAGS) -c -O3 $<
.cpp.o:
  g++  -fopenmp -w -c -O3 -std=c++11 -I$(CUDAINC) $<
  #g++ $(CPPFLAG) -c $<


MODULES=data.f\
upfpsp_module.f\
param_escan.f\
mod_mpi.f\
data_commun_0.f

SRCL=new.f\
vdw.f \
london.f  \
vdwpair.f \
atomic_eggfit.f \
egghead_analytic.f\
egghead_fit_sub.f\
atomic_relaxation_test.f

OBJL=$(SRCL:.f=.o) $(SRC_obj) $(SRC_CG:.f90=.o) $(NVCC_OBJ:.cu=.o) $(CC_OBJ:.c=.o) $(CPP_OBJ:.cpp=.o)

OBJMOD=$(MODULES:.f=.o) $(F90MODULES:.f90=.o)

PWmat: $(OBJMOD) $(OBJL0) $(OBJL) $(MODULES) $(F90MODULES) $(TDDFT_FSRC_OBJ) $(TDDFT_CSRC_OBJ)
  $(LDR) -o $(EXEC_PETOT) $(OBJL0) $(OBJMOD) $(OBJL) $(TDDFT_FSRC_OBJ) $(TDDFT_CSRC_OBJ) $(LIB_ALL) $(PYLIB)
libpwmat: $(OBJMOD) $(OBJL) $(MODULES) $(F90MODULES)
  
