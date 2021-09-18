MPIFC = mpiifort
is_mpiifort := $(shell which mpiifort 1>&2 2> /dev/null; echo $$?)
ifneq ($(is_mpiifort),0)
	MPIFC=mpif90
endif

#FFLAG = -O3 -I/opt/iotk/src/ -mkl
FFLAG = -g -traceback -C -I/opt/iotk/src/ -mkl
#SRCS = $(wildcard *.f90)
#OBJ = $(patsubst %.f90,%.o,$(SRCS))
# priority-sensitive
OBJ = mod_matom.o mod_mpi.o mod_data.o mod_control.o mod_md.o common_module_99.o \
        calc_lin_forceM.o calc_VV_forceM.o calc_NN_forceM.o find_neighbore.o get_expand_MT.o \
        calc_chebyshev.o calc_clebsch_gordan.o calc_U_JM1M2.o calc_loop_ind.o calc_polynomial.o \
	readf_xatom_new.o md.o ranfx.f rantest.f numeric_container.o get_ALI.f gaussj.f factorial.o \
        gen_2b_feature.o gen_3b_feature.o gen_2bgauss_feature.o gen_3bcos_feature.o \
        gen_MTP_feature.o gen_SNAP_feature.o gen_deepMD1_feature.o gen_deepMD2_feature.o \
        find_feature_2b_type1.o find_feature_2b_type2.o find_feature_2b_type3.o \
        find_feature_3b_type1.o find_feature_3b_type2.o find_feature_3b_type3.o \
        find_feature_2bgauss.o find_feature_3bcos.o find_feature_MTP.o \
        find_feature_SNAP.o find_feature_deepMD1.o find_feature_deepMD2.o \
        ML_FF_EF.o main_MD.o \
        read_key_words.o scan_key_words.o transform_to_upper.o \

# priority-sensitive
MODS = calc_lin.mod calc_nn.mod calc_vv.mod common_module_99.mod calc_ftype1.mod calc_2bgauss_feature.mod calc_3bcos_feature.mod calc_ftype2.mod calc_deepmd1_feature.mod calc_deepmd2_feature.mod calc_mtp_feature.mod calc_snap_feature.mod mod_control.mod mod_data.mod mod_parameter.mod mod_md.mod mod_mpi.mod


all: mods main_MD.x
	\cp main_MD.x ../../bin/

main_MD.x: $(OBJ)
	make -C iotk
	\cp iotk/src/libiotk.a libiotk.a
	$(MPIFC) $(FFLAG) libiotk.a -o $@ $(OBJ)

%.o:  %.f90
	$(MPIFC) $(FFLAG) -c $<

calc_lin.mod: calc_lin_forceM.f90
	$(MPIFC) $(FFLAG) -c $<
calc_nn.mod: calc_NN_forceM.f90
	$(MPIFC) $(FFLAG) -c $<
calc_vv.mod: calc_VV_forceM.f90
	$(MPIFC) $(FFLAG) -c $<
common_module_99.mod: common_module_99.f90
	$(MPIFC) $(FFLAG) -c $<
calc_ftype1.mod: gen_2b_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_2bgauss_feature.mod: gen_2bgauss_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_3bcos_feature.mod: gen_3bcos_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_ftype2.mod: gen_3b_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_deepmd1_feature.mod: gen_deepMD1_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_deepmd2_feature.mod: gen_deepMD2_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_mtp_feature.mod: gen_MTP_feature.f90
	$(MPIFC) $(FFLAG) -c $<
calc_snap_feature.mod: gen_SNAP_feature.f90
	$(MPIFC) $(FFLAG) -c $<
mod_control.mod: mod_control.f90
	$(MPIFC) $(FFLAG) -c $<
mod_data.mod: mod_data.f90
	$(MPIFC) $(FFLAG) -c $<
mod_parameter.mod: mod_matom.f90
	$(MPIFC) $(FFLAG) -c $<
mod_md.mod: mod_md.f90
	$(MPIFC) $(FFLAG) -c $<
mod_mpi.mod: mod_mpi.f90
	$(MPIFC) $(FFLAG) -c $<

mods: mod_parameter.mod mod_mpi.mod calc_lin.mod calc_nn.mod calc_vv.mod\
	 common_module_99.mod calc_ftype1.mod\
	 calc_2bgauss_feature.mod calc_3bcos_feature.mod calc_ftype2.mod calc_deepmd1_feature.mod calc_deepmd2_feature.mod calc_mtp_feature.mod calc_snap_feature.mod\
	 mod_control.mod mod_data.mod mod_md.mod
clean:
	make clean -C iotk
	rm -f *.a *.o *.mod *.x 

