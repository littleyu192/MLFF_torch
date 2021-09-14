import os
import parameters as pm
import numpy as np
import numpy as cp
import pandas as pd

def collectAllSourceFiles(workDir=pm.trainSetDir,sourceFileName='MOVEMENT'):
    '''
    搜索工作文件夹，得到所有MOVEMENT文件的路径，并将之存储在pm.sourceFileList中
    
    Determine parameters:
    ---------------------
    pm.sourceFileList:            List对象，罗列了所有MOVEMENT文件的路径        
    '''
    if not os.path.exists(workDir):
        raise FileNotFoundError(workDir+'  is not exist!')
    for path,dirList,fileList in os.walk(workDir):
        if sourceFileName in fileList:
            pm.sourceFileList.append(os.path.abspath(path))


def savePath(featSaveForm='C'):
    '''
    save path to file
    '''
    featSaveForm=featSaveForm.upper()
    pm.numOfSystem=len(pm.sourceFileList)
    with open(pm.fbinListPath,'w') as fbinList:
        fbinList.write(str(pm.numOfSystem)+'\n')
        fbinList.write(str(os.path.abspath(pm.trainSetDir))+'\n')
        for system in pm.sourceFileList:
            fbinList.write(str(system)+'\n')

def combineMovement():
    '''
    combine MOVEMENT file
    '''
    with open(os.path.join(os.path.abspath(pm.trainSetDir),'MOVEMENTall'), 'w') as outfile:     
        # Iterate through list 
        for names in pm.sourceFileList:     
            # Open each file in read mode 
            with open(os.path.join(os.path.abspath(names),'MOVEMENT')) as infile:     
                # read the data from file1 and 
                # file2 and write it in file3 
                outfile.write(infile.read()) 
    
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

def movementUsed():
    '''
    index images not used
    '''
    badImageNum=0
    for names in pm.sourceFileList:
        # image=np.loadtxt(os.path.join(os.path.abspath(names),'info.txt'))
        image=pd.read_csv(os.path.join(os.path.abspath(names),'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
        badimage=image[3:]
        badImageNum=badImageNum+len(badimage)
    
    with open(os.path.join(os.path.abspath(pm.trainSetDir),'imagesNotUsed'), 'w') as outfile:     
        # Iterate through list 
        outfile.write(str(badImageNum)+'  \n')
        index=0
        
        for names in pm.sourceFileList:
            image=np.loadtxt(os.path.join(os.path.abspath(names),'info.txt'))
            badimage=image[3:]
            numOfImage=image[2]
            for i in range(len(badimage)):
                outfile.write(str(badimage[i]+index)+'\n')
            index=index+numOfImage
    


def writeGenFeatInput(UseFtype=pm.use_Ftype):
    
    for ftype in UseFtype:
        if ftype == 1:
            '''
            gen_2b_feature.in
                6.0, 200   !  Rc_M, m_neigh
                2          ! ntype 
                6          ! iat-type 
                5.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                30       ! n3b1, n3b2 
                29          ! iat-type 
                6.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                40       !  n3b1, n3b2 
                0.3    ! E_tolerance (eV)  
                2     ! iflag_ftype:1,2,3 (three different ways for 3b feature, different sin peak spans) 
                1     ! recalc_grid, 0 read from file, 1 recalc 
            '''
            #gen 2b feature input
            with open(pm.Ftype1InputPath,'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'              ! iat-type \n')
                    GenFeatInput.write(str(pm.Ftype1_para['Rc'][i])+','+str(pm.Ftype1_para['Rm'][i])+','+str(pm.Ftype1_para['iflag_grid'][i])+','+str(pm.Ftype1_para['fact_base'][i])+','+\
                        str(pm.Ftype1_para['dR1'][i])+'      !Rc,Rm,iflag_grid,fact_base,dR1 \n')
                    GenFeatInput.write(str(pm.Ftype1_para['numOf2bfeat'][i])+'              ! n2b \n')
                # GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
                GenFeatInput.write(str(pm.Ftype1_para['iflag_ftype'])+'    ! iflag_ftype \n')
                GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')
        if ftype == 2:
            '''
            gen_3b_feature.in
                6.0, 200   !  Rc_M, m_neigh
                2          ! ntype 
                6          ! iat-type 
                5.0,8.0,3.0,2,0.2,0.5,0.5      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 
                2,3       ! n3b1, n3b2 
                29          ! iat-type 
                5.5,10.5,3.0,2,0.2,0.5,0.5      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 
                3,4       !  n3b1, n3b2 
                0.3    ! E_tolerance (eV)  
                2     ! iflag_ftype:1,2,3 (three different ways for 3b feature, different sin peak spans) 
                0     ! recalc_grid, 0 read from file, 1 recalc 
            '''
            #gen 3b feature input
            with open(pm.Ftype2InputPath,'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'          ! iat-type \n')        
                    GenFeatInput.write(str(pm.Ftype2_para['Rc'][i])+','+str(pm.Ftype2_para['Rc2'][i])+','+str(pm.Ftype2_para['Rm'][i])+','+str(pm.Ftype2_para['iflag_grid'][i])+','+str(pm.Ftype2_para['fact_base'][i])+','+\
                        str(pm.Ftype2_para['dR1'][i])+','+str(pm.Ftype2_para['dR2'][i])+'      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 \n')
                    GenFeatInput.write(str(pm.Ftype2_para['numOf3bfeat1'][i])+','+str(pm.Ftype2_para['numOf3bfeat2'][i])+'       ! n3b1, n3b2 \n')
                # GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
                GenFeatInput.write(str(pm.Ftype2_para['iflag_ftype'])+'    ! iflag_ftype \n')
                GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')



def writeFitInput():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')

    natom=200
    m_neigh=pm.maxNeighborNum
    n_image=200
    with open(pm.fitInputPath2,'w') as fitInput:
        fitInput.write(str(len(pm.atomType))+', '+str(natom)+', '+str(m_neigh)+', '+\
                       str(n_image)+'      ! ntype,natom,m_neighb,nimage\n')
        for i in range(pm.atomTypeNum):
            line=str(pm.atomType[i])+', '+str(int(pm.fortranFitFeatNum0[i]))+', '+str(int(pm.fortranFitFeatNum2[i]))+\
                 ', '+str(int(pm.fortranGrrRefNum[i]))+', '+str(float(pm.fortranFitAtomRadii[i]))+', '+\
                 str(pm.fortranFitAtomRepulsingEnergies[i])+'       ! itype, nfeat0,nfeat2,ref_num,rad_atom,wp_atom\n'
            fitInput.write(line)
        fitInput.write(str(pm.fortranGrrKernelAlpha)+', '+str(pm.fortranGrrKernalDist0)+'            ! alpha,dist0 (for kernel)\n')
        fitInput.write(str(pm.fortranFitWeightOfEnergy)+', '+str(pm.fortranFitWeightOfEtot)+', '+str(pm.fortranFitWeightOfForce)+\
                       ', '+str(pm.fortranFitRidgePenaltyTerm)+'        ! E_weight ,Etot_weight, F_weight, delta\n')
                        

def readFeatnum():
    collectAllSourceFiles()
    featnum=0
    for i in pm.use_Ftype:
        with open(os.path.join(pm.sourceFileList[0],'info.txt.Ftype'+str(i)),'r') as sourceFile:
            featnum=featnum+int(sourceFile.readline().split()[0])
    
    pm.realFeatNum=featnum
    pm.nFeats=np.array([pm.realFeatNum,pm.realFeatNum,pm.realFeatNum])
            # pm.fortranFitFeatNum2[i]=pm.fortranFitFeatNum0[i]
    # pm.fortranFitFeatNum0=pm.realFeatNum*np.ones((pm.atomTypeNum,),np.int32)
    # pm.fortranFitFeatNum2=(pm.fortranFitFeatNum0*1.0).astype(np.int32)
    
def calFeatGrid():
    '''
    首先应该从设置文件中读取所有的用户设定

    Determine parameters:
    ---------------------
    mulFactorVectOf2bFeat:    一维pm.mulNumOf2bFeat长度的cp.array,用于计算pm.mulNumOf2bFeat个二体feat的相应参数
    pm.mulFactorVectOf3bFeat:    一维pm.mulNumOf3bFeat长度的cp.array,用于计算pm.mulNumOf3bFeat个三体feat的相应参数
    pm.weightOfDistanceScaler:   标量实数，basic函数中对输入距离矩阵进行Scaler的权重w
    pm.biasOfDistanceScaler：    标量实数，basic函数中对输入距离矩阵进行Scaler的偏置b 
    '''    
    mulFactorVectOf2bFeat={}
    mulFactorVectOf3bFeat1={}
    mulFactorVectOf3bFeat2={}
    h2b={}
    h3b1={}
    h3b2={}
    for itype in range(pm.atomTypeNum):
        mulFactorVectOf2bFeat[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype1_para['numOf2bfeat'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Ftype1_para['Rc'][itype]/2.0
        mulFactorVectOf3bFeat1[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype2_para['numOf3bfeat1'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Ftype2_para['Rc'][itype]/2.0
        mulFactorVectOf3bFeat2[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype2_para['numOf3bfeat2'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Ftype2_para['Rc2'][itype]/2.0
        h2b[itype]=(pm.Ftype1_para['Rc'][itype]-float(mulFactorVectOf2bFeat[itype].max()))
        h3b1[itype]=(pm.Ftype2_para['Rc'][itype]-float(mulFactorVectOf3bFeat1[itype].max()))
        h3b2[itype]=(pm.Ftype2_para['Rc2'][itype]-float(mulFactorVectOf3bFeat2[itype].max()))

        with open(os.path.join(pm.OutputPath,'grid2b_type3.'+str(itype+1)),'w') as f:
            f.write(str(pm.Ftype1_para['numOf2bfeat'][itype])+' \n')
            for i in range(pm.Ftype1_para['numOf2bfeat'][itype]):
                left=mulFactorVectOf2bFeat[itype][i]-h2b[itype]
                right=mulFactorVectOf2bFeat[itype][i]+h2b[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')
        with open(os.path.join(pm.OutputPath,'grid3b_cb12_type3.'+str(itype+1)),'w') as f:
            f.write(str(pm.Ftype2_para['numOf3bfeat1'][itype])+' \n')
            for i in range(pm.Ftype2_para['numOf3bfeat1'][itype]):
                left=mulFactorVectOf3bFeat1[itype][i]-h3b1[itype]
                right=mulFactorVectOf3bFeat1[itype][i]+h3b1[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')
        with open(os.path.join(pm.OutputPath,'grid3b_b1b2_type3.'+str(itype+1)),'w') as f:
            f.write(str(pm.Ftype2_para['numOf3bfeat2'][itype])+' \n')
            for i in range(pm.Ftype2_para['numOf3bfeat2'][itype]):
                left=mulFactorVectOf3bFeat2[itype][i]-h3b2[itype]
                right=mulFactorVectOf3bFeat2[itype][i]+h3b2[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')

# def r_feat_csv(f_feat):
#     """ read feature and energy from pandas data
#     """
#     df   = pd.read_csv(f_feat,dtype=pm.feature_dtype)
#     itypes = df['Type'].values.astype(int)
#     engy = df['dE'].values
#     feat = df.drop(['Type','index','dE','Num','energy'],axis=1).values 
#     # feat = df.iloc[:][4:-1].values
#     engy = engy.reshape([engy.size,1])
#     return itypes,feat,engy

def r_feat_csv(f_feat):
    """ read feature and energy from pandas data
    """
    # df   = pd.read_csv(f_feat,  encoding= 'unicode_escape')
    # import ipdb;ipdb.set_trace()

    df   = pd.read_csv(f_feat,header=None,index_col=False,dtype=pm.feature_dtype, encoding= 'unicode_escape')
    itypes = df[1].values.astype(int)
    engy = df[2].values
    feat = df.drop([0,1,2],axis=1).values 
    engy = engy.reshape([engy.size,1])
    return itypes,feat,engy

def r_egroup_csv(f_egroup):
    """ read feature and energy from pandas data
    """
    df   = pd.read_csv(f_egroup,header=None,index_col=False,dtype=pm.feature_dtype)
    egroup = df[0].values
    divider = df[1].values
    egroup_weight = df.drop([0,1],axis=1).values 
    egroup = egroup.reshape([egroup.size,1])
    divider = divider.reshape([divider.size,1])
    # itypes = df[1].values.astype(int)
    # engy = df[2].values
    # feat = df.drop([0,1,2],axis=1).values 
    # engy = engy.reshape([engy.size,1])
    return egroup,divider,egroup_weight

# def calGrid4()

if __name__ == "__main__":
    pass
