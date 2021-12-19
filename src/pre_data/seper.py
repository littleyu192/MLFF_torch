#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import parameters as pm
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
workpath = os.path.abspath(pm.codedir)
sys.path.append(workpath)
import prepare as pp


def write_egroup_input():
    with open(os.path.join(pm.InputPath, 'egroup.in'), 'w') as f:
        f.writelines(str(pm.dwidth)+'\n')
        f.writelines(str(pm.ntypes)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.b_init[i])+'\n')


def run_write_egroup():
    command = 'write_egroup.x > ./output/out_write_egroup'
    print(command)
    os.system(command)


def write_natoms_dfeat():
    max_natom = int(np.loadtxt(os.path.join(pm.OutputPath, 'max_natom')))
    pp.collectAllSourceFiles()
    f_train_natom = open(pm.f_train_natoms, 'w')
    f_test_natom = open(pm.f_test_natoms, 'w')
    kk = 0
    f_train_dfeat = {}
    f_test_dfeat = {}
    dfeat_names = {}
    for i in pm.use_Ftype:
        f_train_dfeat[i] = open(pm.f_train_dfeat+str(i), 'w')
        f_test_dfeat[i] = open(pm.f_test_dfeat+str(i), 'w')
        feat_head_tmp = pd.read_csv(os.path.join(
            pm.trainSetDir, 'trainData.txt'+'.Ftype'+str(i)), header=None).values[:, :3]
        feat_tmp = pd.read_csv(os.path.join(
            pm.trainSetDir, 'trainData.txt'+'.Ftype'+str(i)), header=None).values[:, 4:].astype(float)
        dfeat_names[i] = pd.read_csv(os.path.join(
            pm.trainSetDir, 'inquirepos'+str(i)+'.txt'), header=None).values[:, 1:].astype(int)
        if kk == 0:
            feat = feat_tmp
        else:
            # import ipdb;ipdb.set_trace()
            feat = np.concatenate((feat, feat_tmp), axis=1)
        
        kk = kk+1
    feat_all = np.concatenate((feat_head_tmp, feat), axis=1)
    # import ipdb;ipdb.set_trace()
    egroup_all = pd.read_csv(os.path.join(
        pm.trainSetDir, 'Egroup_weight'), header=None, names=range(max_natom+2))
    egroup_all = egroup_all.fillna(0)
    egroup_all = egroup_all.values[:, :].astype(float)

    count = 0
    Imgcount = 0
    feat_train = np.empty([0, feat_all.shape[1]])
    feat_test = np.empty([0, feat_all.shape[1]])
    egroup_train = np.empty([0, egroup_all.shape[1]])
    egroup_test = np.empty([0, egroup_all.shape[1]])
    for system in pm.sourceFileList:

        infodata = pd.read_csv(os.path.join(system, 'info.txt.Ftype'+str(
            pm.use_Ftype[0])), header=None, delim_whitespace=True).values[:, 0].astype(int)
        natom = infodata[1]
        ImgNum = infodata[2]-(len(infodata)-3)
        trainImgNum = int(ImgNum*(1-pm.test_ratio))
        trainImg = np.arange(0, trainImgNum)
        testImg = np.arange(trainImgNum, ImgNum)

        for i in trainImg:
            f_train_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            for mm in pm.use_Ftype:
                f_train_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(
                    mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i), 1])+'\n')
                    
        for i in testImg:
            f_test_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            for mm in pm.use_Ftype:
                f_test_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(
                    mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i), 1])+'\n')
        feat_train = np.concatenate(
            (feat_train, feat_all[count:(count+natom*len(trainImg)), :]), axis=0)
        feat_test = np.concatenate(
            (feat_test, feat_all[(count+natom*len(trainImg)):(count+natom*ImgNum), :]), axis=0)

        egroup_train = np.concatenate(
            (egroup_train, egroup_all[count:(count+natom*len(trainImg)), :]), axis=0)
        egroup_test = np.concatenate((egroup_test, egroup_all[(
            count+natom*len(trainImg)):(count+natom*ImgNum), :]), axis=0)

        count = count+natom*ImgNum
        Imgcount = Imgcount+ImgNum
    # import ipdb;ipdb.set_trace()
    np.savetxt(pm.f_train_feat, feat_train, delimiter=',')
    np.savetxt(pm.f_test_feat, feat_test, delimiter=',')
    np.savetxt(pm.f_train_egroup, egroup_train, delimiter=',')
    np.savetxt(pm.f_test_egroup, egroup_test, delimiter=',')

    f_train_natom.close()
    f_test_natom.close()
    for i in pm.use_Ftype:
        f_train_dfeat[i].close()
        f_test_dfeat[i].close()


def write_dR_neigh():
    # 需要生成一个自己的info文件 先用gen2b的代替
    infodata = pd.read_csv(os.path.join(pm.sourceFileList[0],'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
    natom = infodata[1]
    img_num = infodata[2] - (len(infodata)-3)
    train_img_num = int(img_num * (1 - pm.test_ratio))
    dR_neigh = pd.read_csv(pm.dRneigh_path, header=None, delim_whitespace=True)
    if img_num * pm.maxNeighborNum * natom != dR_neigh.shape[0]:
        raise ValueError("dim not match")
    index = train_img_num * natom * pm.maxNeighborNum
    train_img = dR_neigh[:index]
    test_img = dR_neigh[index:]
    train_img.to_csv(pm.f_train_dR_neigh, header=False, index=False)
    test_img.to_csv(pm.f_test_dR_neigh, header=False, index=False)


if __name__ == '__main__':
    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    write_egroup_input()
    run_write_egroup()
    write_natoms_dfeat()
    if (pm.dR_neigh):
        write_dR_neigh()
