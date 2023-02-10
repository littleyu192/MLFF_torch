ALLEXE = calc_lin_force_Etot.r  feat_PCA_metric2.r   linear_force_totE.r  sparcify_CUR2.r \
         calc_E_forceM.r calc_lin_forceM.r GPR_fit_forceM.r linear_forceM.r \
		 linear_clusterM.r calc_lin_cluster.r
NORANEXE =  calc_lin_force_Etot.r  linear_force_totE.r  \
         calc_E_forceM.r calc_lin_forceM.r GPR_fit_forceM.r linear_forceM.r \
		 linear_clusterM.r calc_lin_cluster.r
RANEXE = feat_collect_PCA.r
NORANOBJ = $(NORANEXE:.r=.o)
RANOBJ = $(RANEXE:.r=.o)
.PHONY: all lin gpr

all: $(ALLEXE)

	
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


lin: all
	make lin -Cfread_dfeat
gpr: all
	make gpr -Cfread_dfeat
