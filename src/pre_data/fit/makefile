ALLEXE = calc_lin_forceMM.r  feat_collect_PCA.r  fit_lin_forceMM.r select_mm_VV.r fit_VV_forceMM.r calc_VV_forceMM.r fit_vdw.r
NORANEXE =  calc_lin_forceMM.r  feat_collect_PCA.r  fit_lin_forceMM.r fit_VV_forceMM.r calc_VV_forceMM.r fit_vdw.r
RANEXE = feat_collect_PCA.r select_mm_VV.r
NORANOBJ = $(NORANEXE:.r=.o)
RANOBJ = $(RANEXE:.r=.o)
#.PHONY: all lin 
.PHONY: all

MPIFC = mpiifort
is_mpiifort := $(shell which mpiifort 1>&2 2> /dev/null; echo $$?)
ifneq ($(is_mpiifort),0)
	MPIFC=mpif90
endif


all: $(ALLEXE) select_mm_VV_forceMPI.r feat_dist_xp.r
	\cp *.r ../../bin/

$(NORANEXE): %.r : %.o
	ifort -o $@ $< -mkl
$(NORANOBJ): %.o : %.f
	ifort -c $<
$(RANEXE): %.r : %.o ran1.o
	ifort -o $@ $< ran1.o -mkl
$(RANOBJ): %.o : %.f
	ifort -c $<
ran1.o: ran1.f
	ifort -c ran1.f

select_mm_VV_forceMPI.r:
	$(MPIFC) -O3 select_mm_VV_forceMPI.f  calc_VV_force_sub.f fit_VV_force_sub.f -mkl -o $@
feat_dist_xp.r:
	ifort -O3 feat_dist_xp.f -mkl -o $@

#lin: all
#	make lin -C fread_dfeat

clean:
	rm -f *.o *.r *.mod
	rm -f ../../bin/calc_lin_forceMM.r
	rm -f ../../bin/feat_collect_PCA.r
	rm -f ../../bin/fit_lin_forceMM.r
	rm -f ../../bin/fit_VV_forceMM.r
	rm -f ../../bin/select_mm_VV.r
	rm -f ../../bin/calc_VV_forceMM.r
	rm -f ../../bin/feat_dist_xp.r
	rm -f ../../bin/fit_vdw.r
	rm -f ../../bin/select_mm_VV_forceMPI.r
