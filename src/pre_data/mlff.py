#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import parameters as pm
#pm.fortranFitSourceDir = codepath + '/fit'
import prepare as pp
import fortran_fitting as ff
import pandas as pd

if os.path.exists('./input/'):
    pass
else:
    os.mkdir('./input/')
    os.mkdir('./output')

# if True:
#     # liuliping auto vdw_fitB.ntype
#     print('auto creating vdw_fitB.ntype, donot use your own vdw_fitB.ntype file')
#     print('please modify parameters.py to specify your vdw parameters')
#     if not os.path.exists(pm.fitModelDir):
#         os.makedirs(pm.fitModelDir)
#     strength_rad = 0.0
#     if pm.isFitVdw == True:
#         strength_rad = 1.0
#     vdw_input = {
#         'ntypes': pm.ntypes,
#         'nterms': 1,
#         'atom_type': pm.atomType,
#         'rad': [strength_rad for i in range(pm.ntypes)],
#         'e_ave': [500.0 for i in range(pm.ntypes)],
#         'wp': [ [0.8 for i in range(pm.ntypes*1)] for i in range(pm.ntypes)]
#     }
#     if hasattr(pm, 'vdwInput'):
#         vdw_input = pm.vdwInput
#     pp.writeVdwInput(pm.fitModelDir, vdw_input)

if True:
    print('auto creating vdw_fitB.ntype, donot use your own vdw_fitB.ntype file')
    print('please modify parameters.py to specify your vdw parameters')
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)
    strength_rad = 0.0
    if pm.isFitVdw == True:
        strength_rad = 1.0
    vdw_input = {
        'ntypes': pm.ntypes,
        'nterms': 1,
        'atom_type': pm.atomType,
        'rad': [strength_rad for i in range(pm.ntypes)],
        'e_ave': [0.0 for i in range(pm.ntypes)],
        'wp': [ [0.0 for i in range(pm.ntypes*1)] for i in range(pm.ntypes)]
    }
    if hasattr(pm, 'vdwInput'):
        vdw_input = pm.vdwInput
    pp.writeVdwInput(pm.fitModelDir, vdw_input)

if pm.isCalcFeat:
    if os.path.exists(os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.Ftype1')):
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.*'))
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'inquirepos*'))
        os.system('rm ' + pm.dRneigh_path)
    else:
        pass
    if os.path.exists(os.path.join(pm.trainSetDir, 'lppData.txt')):
        os.system('rm '+os.path.join(pm.trainSetDir, 'lppData.txt'))
    else:
        pass
    pp.collectAllSourceFiles()
    pp.savePath()
    pp.combineMovement()
    pp.writeGenFeatInput()
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')

    calFeatGrid = False
    for i in range(pm.atomTypeNum):
        if pm.Ftype1_para['iflag_grid'][i] == 3 or pm.Ftype2_para['iflag_grid'][i] == 3:
            calFeatGrid = True
    if calFeatGrid:
        pp.calFeatGrid()
    for i in pm.use_Ftype:
        # import ipdb;ipdb.set_trace()
        command = pm.Ftype_name[i]+".x > ./output/out"+str(i)
        print(command)
        os.system(command)
    # import ipdb;ipdb.set_trace()
else:
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    pp.writeGenFeatInput()
    pp.collectAllSourceFiles()

if pm.isFitVdw:
    ff.makeFitDirAndCopySomeFiles()
    ff.copyData()
    ff.writeFitInput()
    ff.FeatCollectIn()
    # command = 'feat_collect_PCA.r feat_collect.in; '
    # current_dir = os.getcwd()
    # os.chdir(pm.fitModelDir)
    # os.system(command)
    # os.chdir(current_dir)
    # # command = 'feat_collect_PCA.r feat_collect.in; ' + 'fit_vdw.r feat_PV.1; '
    # command = 'fit_vdw.r feat_PV.1; '
    # os.chdir(pm.fitModelDir)
    # os.system(command)
    # os.chdir(current_dir)
    current_dir = os.getcwd()

    command1 = 'feat_collect_PCA.r feat_collect.in; '
    command2 = 'fit_vdw.r feat_PV.1; '
    os.chdir(pm.fitModelDir)
    print(command1)
    # import ipdb; ipdb.set_trace()
    os.system(command1)
    print(command2)
    os.system(command2)
    os.chdir(current_dir)

if pm.isFitLinModel:
    ff.fit()

if pm.isRunMd100:
    os.environ["CUDA_VISIBLE_DEVICES"] = pm.cuda_dev
    # import preparatory_work as ppw
    sys.path.append(codepath+'/../pre_data')
    from md_run100 import MdRunner
    # if not pm.isCalcFeat:
    #     ppw.preparatoryWorkForMdImage()
    mdRunner=MdRunner()
    movementPath=os.path.join(pm.mdImageFileDir,'MOVEMENT')
    with open(movementPath,'r') as sourceFile:
        allData=sourceFile.read()
        nimage=allData.count('Iteration')
        # for imageIndex in range(nimage):
            
    for i in range(nimage):
        mdRunner.run100(i)
    mdRunner.final()

# liuliping MD_code.NEW, md100 workflow
if hasattr(pm, 'isNewMd100'):
    if pm.isNewMd100:
        os.system('rm -f MOVEMENT')
        sys.path.append(codepath+'/../test')
        import md100
        num_process = 1                  #use 1 mpi process as default
        
        if hasattr(pm, 'imodel'):
            imodel = pm.imodel
        else:
            print("Specification of parameter imodel is required!")
        if hasattr(pm, 'md_num_process'):
            num_process = pm.md_num_process
        md100.run_md100(imodel=imodel, atom_type=pm.atomType, num_process=num_process)

# if pm.dR_neigh:
#     movement_path = os.path.join(pm.trainSetDir, "MOVEMENT")
#     cmd1 = "sed -n '/force/,+108p' " + movement_path + " > tmp.txt"
#     cmd2 = "sed '/force/d' tmp.txt | awk '{print $2,$3,$4}' OFS=',' > Force.txt"

#     os.system(cmd1)
#     os.system(cmd2)

#     force = pd.read_csv("Force.txt", header=None)
#     os.system("rm Force.txt tmp.txt")
#     force_path = os.path.join(pm.trainSetDir, "force.csv")
#     force.to_csv(force_path, header=False, index=False)
