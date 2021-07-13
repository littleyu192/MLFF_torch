import os
import sys
import parameters as pm
codepath = os.getcwd()
pm.fortranFitSourceDir = codepath + '/data/fit'
import prepare as pp
import fortran_fitting as ff
# genFeatInputFile='./gen_feature.in'
if os.path.exists('./input/'):
    pass
else:
    os.mkdir('./input/')
    os.mkdir('./output')



if pm.isCalcFeat:
    if os.path.exists(os.path.join(os.path.abspath(pm.trainSetDir),'trainData.txt.Ftype1')):
        os.system('rm '+os.path.join(os.path.abspath(pm.trainSetDir),'trainData.txt.*'))
        os.system('rm '+os.path.join(os.path.abspath(pm.trainSetDir),'inquirepos*'))
    else:
        pass
    if os.path.exists(os.path.join(pm.trainSetDir,'lppData.txt')):
        os.system('rm '+os.path.join(pm.trainSetDir,'lppData.txt'))
    else:
        pass
    pp.collectAllSourceFiles()
    pp.savePath()
    pp.combineMovement()
    pp.writeGenFeatInput()
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    
    calFeatGrid=False

    for i in range(pm.atomTypeNum):
        if pm.Ftype1_para['iflag_grid'][i] == 3 or pm.Ftype2_para['iflag_grid'][i] == 3:
            calFeatGrid=True
    if calFeatGrid:
        pp.calFeatGrid()
    for i in pm.use_Ftype:
        command=pm.Ftype_name[i]+".x > ./output/out"+str(i)
        print(command)
        os.system(command)
else:
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    pp.writeGenFeatInput()
    pp.collectAllSourceFiles()

if pm.isFitVdw:
    ff.makeFitDirAndCopySomeFiles()
    # readFittingParameters()
    ff.copyData()
    ff.writeFitInput()
    ff.FeatCollectIn()
    # liuliping: deprecate makefile, use bash commands
    #command='make pca -C'+pm.fitModelDir
    command = 'feat_collect_PCA.r feat_collect.in; '
    current_dir = os.getcwd()
    os.chdir(pm.fitModelDir)
    os.system(command)
    os.chdir(current_dir)
    # liuliping: deprecate makefile, use bash commands
    #command='make vdw -C'+pm.fitModelDir
    command = 'feat_collect_PCA.r feat_collect.in; ' + 'fit_vdw.r feat_PV.1; '
    os.chdir(pm.fitModelDir)
    os.system(command)
    os.chdir(current_dir)

if pm.isFitLinModel:
    ff.fit()
