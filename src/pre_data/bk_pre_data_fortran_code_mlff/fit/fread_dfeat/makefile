
.PHONY: none lin clst all pca vdw

none:

all:
	make all -C..

lin: energyL.pred.1 forceL.pred.1

clst: energyC.pred.1 forceC.pred.1

pca: feat_PV.1

vdw: vdw_fitB.ntype

energyC.pred.1: ../calc_lin_cluster.r cluster_fitB.ntype
	../calc_lin_cluster.r
cluster_fitB.ntype: ../linear_clusterM.r feat_PV.1
	../linear_clusterM.r
energyL.pred.1: ../calc_lin_forceMM.r linear_fitB.ntype
	../calc_lin_forceMM.r
linear_fitB.ntype: ../fit_lin_forceMM.r feat_PV.1
	../fit_lin_forceMM.r
feat_PV.1: ../feat_collect_PCA.r feat_collect.in
	../feat_collect_PCA.r
vdw_fitB.ntype: ../fit_vdw.r feat_PV.1
	../fit_vdw.r
	
