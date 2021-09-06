#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.lib.type_check import real
import os
import sys
sys.path.append(os.getcwd())
import parameters as pm
import prepare
import numpy as np
import pandas as pd
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
from read_all import read_allnn
import torch

if pm.torch_dtype == 'float32':
    torch_dtype = torch.float32
    from convert_dfeat import convert_dfeat
else:
    torch_dtype = torch.float64
    from convert_dfeat64 import convert_dfeat

class MinMaxScaler:
    ''' a*x +b = x_scaled like sklearn's MinMaxScaler
        note cp.atleast_2d and self.a[xmax==xmin] = 0
    '''

    def __init__(self, feature_range=(0, 1)):
        self.fr = feature_range
        self.a = 0
        self.b = 0

    def fit_transform(self, x):

        if len(x) == 0:
            self.a = 0
            self.b = 0
            return x
        x = np.atleast_2d(x)
        xmax = x.max(axis=0)
        xmin = x.min(axis=0)
        self.a = (self.fr[1] - self.fr[0]) / (xmax-xmin)
        self.a[xmax-xmin <= 0.1] = 10

        self.a[xmax < 0.01] = 1
        self.a[xmax == xmin] = 0  # important !!!
        self.b = self.fr[0] - self.a*xmin
        return self.transform(x)

    def transform(self, x):
        x = np.atleast_2d(x)
        return self.a*x + self.b

    def inverse_transform(self, y):
        y = np.atleast_2d(y)
        return (y - self.b) / self.a


class DataScaler:
    def __init__(self):
        self.feat_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feat_a = None
        self.engy_scaler = MinMaxScaler(feature_range=(0, 1))
        self.engy_a = None

    #===========================================================================

    def get_scaler(self, f_feat, f_ds, b_save=True):
        itypes,feat,engy = prepare.r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        
        _ = self.feat_scaler.fit_transform(feat)
        _ = self.engy_scaler.fit_transform(engy)
        
        feat_b      = self.feat_scaler.transform(np.zeros((1, feat.shape[1])))    
        self.feat_a = self.feat_scaler.transform(np.ones((1, feat.shape[1]))) - feat_b
        engy_b      = self.engy_scaler.transform(0)
        self.engy_a = self.engy_scaler.transform(1) - engy_b

        #return self.feat_scaler

    #===========================================================================
    
    def pre_feat(self, feat):
        return self.feat_scaler.transform(feat)


class DataScalers:
    '''
    The wrapper for multiple elements. Generally a dictionary with data scalers for each element.
    Notice the 's'. It is important.
    '''

    def __init__(self, f_ds, f_feat, load=False):

        self.scalers = {}
        self.feat_as = {}
        self.engy_as = {}
        
        for i in range(pm.ntypes):
            self.scalers[pm.atomType[i]] = DataScaler()

        from os import path

        if load and path.isfile(f_ds):
            self.loadDSs_np(f_ds)
        elif path.isfile(f_feat):
            self.get_scalers(f_feat, f_ds, b_save=True)
        else:
            exit(["===Error in DataScaler, don't find ", f_ds, f_feat, '==='])

        # return

    def get_scalers(self, f_feat, f_ds, b_save=True):
        itypes,feat,engy = prepare.r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        print('=DS.get_scaler ', f_feat, 'engy.shape, feat.dtype', engy.shape, engy.dtype)
        print('=DS.get_scaler ', f_feat, 'itypes.shape, feat.dtype', itypes.shape, itypes.dtype)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            subfeat = feat[itypes == itype]
            subengy = engy[itypes == itype]
            _ = self.scalers[itype].feat_scaler.fit_transform(subfeat)
            _ = self.scalers[itype].engy_scaler.fit_transform(subengy)
            feat_b = self.scalers[itype].feat_scaler.transform(np.zeros((1, subfeat.shape[1])))
            engy_b = self.scalers[itype].engy_scaler.transform(np.zeros((1, subengy.shape[1])))
            self.feat_as[itype] = self.scalers[itype].\
                                 feat_scaler.transform(np.ones((1, subfeat.shape[1]))) - feat_b
            self.engy_as[itype] = self.scalers[itype].\
                                 engy_scaler.transform(np.ones((1, subengy.shape[1]))) - engy_b

        if b_save:
            self.save2np(f_ds)
    
        #return self.feat_scalers
    def pre_engy(self, engy, itypes):
            # engy_scaled = cp.zeros_like(engy)
            # for i in range(pm.ntypes):
            #     itype = pm.atomType[i]
            #     engy_scaled[itypes == itype] = self.scalers[itype].\
            #         engy_scaler.transform(engy[itypes == itype])
        return engy

    def pre_feat(self, feat, itypes):
        feat_scaled = np.zeros(feat.shape)
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaled[itypes == itype] = self.scalers[itype].\
                                           feat_scaler.transform(feat[itypes == itype])
        return feat_scaled

    def pre_dfeat(self, dfeat, itypes, nblt):
        dfeat_scaled = np.zeros(dfeat.shape)
        natoms=dfeat.shape[0]
        max_nb=dfeat.shape[1]
        featnum=dfeat.shape[2]
       
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            dfeat_scaled[(itypes.squeeze()[nblt-1] == itype) & (nblt>0)] = self.feat_as[itype][:,:,np.newaxis]\
                                            *dfeat[(itypes.squeeze()[nblt-1]==itype)&(nblt>0)]
        return dfeat_scaled


    def save2np(self, f_npfile):
        dsnp = []
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaler = self.scalers[itype].feat_scaler
            engy_scaler = self.scalers[itype].engy_scaler
            dsnp.append(np.array(feat_scaler.fr))
            dsnp.append(np.array(feat_scaler.a))
            dsnp.append(np.array(feat_scaler.b))
            dsnp.append(np.array(self.feat_as[itype]))
            dsnp.append(np.array(engy_scaler.fr))
            dsnp.append(np.array(engy_scaler.a))
            dsnp.append(np.array(engy_scaler.b))
            dsnp.append(np.array(self.engy_as[itype]))
        dsnp = np.array(dsnp)
        np.save(f_npfile, dsnp)
        print('DataScaler.save2np to', f_npfile, dsnp.dtype, dsnp.shape)
        return

    def loadDSs_np(self, f_npfile):
        dsnp = np.load(f_npfile, allow_pickle=True)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            self.scalers[itype].feat_scaler.fr = np.asarray(dsnp[8*i+0])
            self.scalers[itype].feat_scaler.a  = np.asarray(dsnp[8*i+1])
            self.scalers[itype].feat_scaler.b  = np.asarray(dsnp[8*i+2])
            self.feat_as[itype]         = np.asarray(dsnp[8*i+3])
            self.scalers[itype].engy_scaler.fr = np.asarray(dsnp[8*i+4])
            self.scalers[itype].engy_scaler.a  = np.asarray(dsnp[8*i+5])
            self.scalers[itype].engy_scaler.b  = np.asarray(dsnp[8*i+6])
            self.engy_as[itype]         = np.asarray(dsnp[8*i+7])

        print('DataScaler.loadDS_np from', f_npfile, dsnp.dtype, dsnp.shape)
        #for i in range(dsnp.shape[0]):
        #    print("dsnp[i]",i, dsnp[i].shape)
        return


def process_data(f_train_feat, f_train_dfeat, f_train_natoms, f_train_egroup,
                 scalers, nn_data_path):
    if not os.path.exists(nn_data_path):
        os.makedirs(nn_data_path)
    # natoms contain all atomnum of each image, format: totnatom, type1n, type2 n
    natoms = np.loadtxt(f_train_natoms, dtype=int)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(f_train_natoms)
    print(natoms.shape)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    nImg = natoms.shape[0]
    indImg = np.zeros((nImg+1,), dtype=int)
    indImg[0] = 0
    for i in range(nImg):
        indImg[i+1] = indImg[i] + natoms[i, 0]

    itypes, feat, engy = prepare.r_feat_csv(f_train_feat)
    feat_scaled = scalers.pre_feat(feat, itypes)
    engy_scaled = scalers.pre_engy(engy, itypes)
    # feat_scaled = pre_feat(scalers, feat, itypes)
    # engy_scaled = engy
    egroup, divider, egroup_weight = prepare.r_egroup_csv(f_train_egroup)
    if os.path.exists(os.path.join(pm.dir_work, 'weight_for_cases')):
        weight_all = pd.read_csv(os.path.join(pm.dir_work, 'weight_for_cases'),
                                 header=None, encoding= 'unicode_escape').values[:, 0].astype(pm.torch_dtype).reshape(-1, 1)
    else:
        weight_all = np.ones((engy_scaled.shape[0], 1))
    nfeat0m = feat_scaled.shape[1]  # 每个原子特征的维度
    itype_atom = np.asfortranarray(np.array(pm.atomType).transpose())  # 原子类型

    feat_scale_a = np.zeros((nfeat0m, pm.ntypes))
    for i in range(pm.ntypes):
        itype = pm.atomType[i]
        feat_scale_a[:, i] = scalers.scalers[itype].feat_scaler.a
        feat_scale_a = np.asfortranarray(feat_scale_a)  # scaler 的 a参数
    # feat_scale_a=np.ones((nfeat0m,pm.ntypes))   #如果不做scale，赋值为1
    # feat_scale_a = np.asfortranarray(feat_scale_a)
    init = pm.use_Ftype[0]

    dfeatdirs = {}
    energy_all = {}
    force_all = {}
    num_neigh_all = {}
    list_neigh_all = {}
    iatom_all = {}
    dfeat_tmp_all = {}
    num_tmp_all = {}
    iat_tmp_all = {}
    jneigh_tmp_all = {}
    ifeat_tmp_all = {}
    nfeat = {}
    nfeat[0] = 0
    flag = 0
    # 读取 dfeat file
    for m in pm.use_Ftype:
        dfeatdirs[m] = np.unique(pd.read_csv(
            f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values[:, 0])
        for k in dfeatdirs[m]:
            read_allnn.read_dfeat(k, itype_atom, feat_scale_a, nfeat[flag])
            if flag == 0:
                energy_all[k] = np.array(
                    read_allnn.energy_all).astype(pm.torch_dtype)
                force_all[k] = np.array(read_allnn.force_all).transpose(
                    1, 0, 2).astype(pm.torch_dtype)
                list_neigh_all[k] = np.array(
                    read_allnn.list_neigh_all).transpose(1, 0, 2).astype(int)
                iatom_all[k] = np.array(read_allnn.iatom)

            nfeat[flag+1] = np.array(read_allnn.feat_all).shape[0]
            dfeat_tmp_all[k] = np.array(read_allnn.dfeat_tmp_all).astype(float)
            num_tmp_all[k] = np.array(read_allnn.num_tmp_all).astype(int)
            iat_tmp_all[k] = np.array(read_allnn.iat_tmp_all).astype(int)
            jneigh_tmp_all[k] = np.array(read_allnn.jneigh_tmp_all).astype(int)
            ifeat_tmp_all[k] = np.array(read_allnn.ifeat_tmp_all).astype(int)
            read_allnn.deallo()
        flag = flag+1
    #pm.fitModelDir=./fread_dfeat  
    with open(os.path.join(pm.fitModelDir, "feat.info"), 'w') as f:
        print(os.path.join(pm.fitModelDir, "feat.info"))
        f.writelines(str(pm.iflag_PCA)+'\n')
        f.writelines(str(len(pm.use_Ftype))+'\n')
        for m in range(len(pm.use_Ftype)):
            f.writelines(str(pm.use_Ftype[m])+'\n')

        f.writelines(str(pm.ntypes)+', '+str(pm.maxNeighborNum)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.atomType[i])+'  ' +
                         str(nfeat0m)+'  '+str(nfeat0m)+'\n')
        for i in range(pm.ntypes):
            for m in range(len(pm.use_Ftype)):
                f.writelines(str(nfeat[m+1])+'  ')
            f.writelines('\n')
    dfeat_names = {}
    image_nums = {}
    pos_nums = {}
    for m in pm.use_Ftype:
        values = pd.read_csv(f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values
        dfeat_names[m] = values[:, 0]
        image_nums[m] = values[:, 1].astype(int)
        pos_nums[m] = values[:, 2].astype(int)
        nImg = image_nums[m].shape[0]

    fors_scaled = []
    nblist = []
    for ll in range(len(image_nums[init])):
        fors_scaled.append(
            force_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
        nblist.append(
            list_neigh_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
    fors_scaled = np.concatenate(fors_scaled, axis=0)
    nblist = np.concatenate(nblist, axis=0)
    
# ========================================================================
    img_num = indImg.shape[0] - 1
    convert_dfeat.allo(nfeat0m, indImg[-1], pm.maxNeighborNum)
    for i in range(img_num):
        dfeat_name={}
        image_num={}
        for mm in pm.use_Ftype:
            dfeat_name[mm] = dfeat_names[mm][i]
            image_num[mm] = image_nums[mm][i]
        kk=0
        for mm in pm.use_Ftype:
            dfeat_tmp=np.asfortranarray(dfeat_tmp_all[dfeat_name[mm]][:,:,image_num[mm]-1])
            jneigh_tmp=np.asfortranarray(jneigh_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            ifeat_tmp=np.asfortranarray(ifeat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            iat_tmp=np.asfortranarray(iat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            convert_dfeat.conv_dfeat(image_num[mm],nfeat[kk],indImg[i],num_tmp_all[dfeat_name[mm]][image_num[mm]-1],dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)
            kk=kk+1

    dfeat_scaled = np.array(convert_dfeat.dfeat).transpose(1,2,0,3).astype(pm.torch_dtype)
    convert_dfeat.deallo()
    print("feat_scaled shape" + str(feat_scaled.shape))
    print("fors_scaled shape" + str(fors_scaled.shape))
    print("nblist shape" + str(nblist.shape))
    print("engy_scaled shape" + str(engy_scaled.shape))
    print("itypes shape" + str(itypes.shape))
    print("egroup_weight shape" + str(egroup_weight.shape))
    print("weight_all shape" + str(weight_all.shape))
    print("egroup shape" + str(egroup.shape))
    print("divider shape" + str(egroup.shape))
    print("dfeat_scaled shape" + str(dfeat_scaled.shape))
    np.save(nn_data_path + "/feat_scaled.npy", feat_scaled)
    np.save(nn_data_path + "/fors_scaled.npy", fors_scaled)
    np.save(nn_data_path + "/nblist.npy", nblist)
    np.save(nn_data_path + "/engy_scaled.npy", engy_scaled)
    np.save(nn_data_path + "/itypes.npy", itypes)
    np.save(nn_data_path + "/egroup_weight.npy", egroup_weight)
    np.save(nn_data_path + "/weight_all.npy", weight_all)
    np.save(nn_data_path + "/egroup.npy", egroup)
    np.save(nn_data_path + "/divider.npy", divider)
    np.save(nn_data_path + "/dfeat_scaled.npy", dfeat_scaled)
    np.save(nn_data_path + "/ind_img.npy", np.array(indImg).reshape(-1))


def main():
    # 计算scale变换的参数
    data_scalers = DataScalers(f_ds=pm.f_data_scaler,
                                   f_feat=pm.f_train_feat)
    
    # scalers_train = get_scalers(pm.f_train_feat, pm.f_data_scaler, True)
    read_allnn.read_wp(pm.fitModelDir, pm.ntypes)
    print(read_allnn.wp_atom)
    process_data(pm.f_train_feat,
                 pm.f_train_dfeat,
                 pm.f_train_natoms,
                 pm.f_train_egroup, 
                 data_scalers,
                 pm.train_data_path)
    # scalers_test = get_scalers(pm.f_test_feat, pm.f_data_scaler, False)
    data_scalers = DataScalers(f_ds=pm.f_data_scaler,
                                   f_feat=pm.f_test_feat)
    process_data(pm.f_test_feat,
                 pm.f_test_dfeat,
                 pm.f_test_natoms,
                 pm.f_test_egroup,
                 data_scalers,
                 pm.test_data_path)

if __name__ == '__main__':
    main()
