#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import parameters as pm
pm.fortranFitSourceDir = codepath + '/fit'
import prepare as pp
import fortran_fitting as ff

if os.path.exists('./input/'):
    pass
else:
    os.mkdir('./input/')
    os.mkdir('./output')


if pm.isCalcFeat:
    if os.path.exists(os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.Ftype1')):
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.*'))
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'inquirepos*'))
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
        command = pm.Ftype_name[i]+".x > ./output/out"+str(i)
        print(command)
        os.system(command)
else:
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    pp.writeGenFeatInput()
    pp.collectAllSourceFiles()

if pm.isFitVdw:
    ff.makeFitDirAndCopySomeFiles()
    ff.copyData()
    ff.writeFitInput()
    ff.FeatCollectIn()
    command = 'feat_collect_PCA.r feat_collect.in; '
    current_dir = os.getcwd()
    os.chdir(pm.fitModelDir)
    os.system(command)
    os.chdir(current_dir)
    command = 'feat_collect_PCA.r feat_collect.in; ' + 'fit_vdw.r feat_PV.1; '
    os.chdir(pm.fitModelDir)
    os.system(command)
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
        imodel = 3    # 1:linear;  2:VV;   3:NN;
        num_process = 1
        if hasattr(pm, 'md100_imodel'):
            imodel = pm.md100_imodel
        if hasattr(pm, 'md_num_process'):
            num_process = pm.md_num_process
        md100.run_md100(imodel=imodel, atom_type=pm.atomType, num_process=num_process)

