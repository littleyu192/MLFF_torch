import numpy as np
import os

isCalcFeat=True
isFitLinModel=False
isClassify=False
isRunMd=False                                   #是否训练运行md  default:False
isRunMd_nn=False
isFollowMd=False                                #是否是接续上次的md继续运行  default:False
isFitVdw=True
isRunMd100_nn=False
isRunMd100=False
add_force=False     # for NN md
#************** Dir **********************
prefix = r'./'
trainSetDir = r'./PWdata'
codedir=r'/home/husiyu/software/MLFF_torch'
fortranFitSourceDir=codedir+'/src/pre_data/fit/'
fitModelDir = r'./fread_dfeat'
train_data_path = r'./train_data/final_train'
test_data_path = r'./train_data/final_test'
genFeatDir = r'./gen_feature'
test_ratio = 0.2

mdImageFileDir=r'./MD'                              #设置md的初始image的文件所在的文件夹  default:'.'

#pbc = True
isCalcFeat=True
isFitLinModel=True
#isClassify=True
#isRunMd=True                                   #是否训练运行md  default:False
#isRunMd_nn=True
#isFollowMd=True                                #是否是接续上次的md继续运行  default:False
#isFitVdw=True
#isRunMd100_nn=True
#isRunMd100=True
#add_force=True     # for NN md
#********* for gen_feature.in *********************
atomType=[3]  
maxNeighborNum=100
natoms=[230]
iflag_PCA=1
Rc_M=6.1                     # max of Rcut

Ftype_name={1:'gen_2b_feature',2:'gen_3b_feature'}
# Ftype2_name='gen_3b_feature'
use_Ftype=[1,2]
nfeat_type=len(use_Ftype)
Ftype1_para={
    'numOf2bfeat':[36],       # [itpye1,itype2]
    'Rc':[6.1],
    'Rm':[5.0],
    'iflag_grid':[3],                      # 1 or 2 or 3
    'fact_base':[0.2],
    'dR1':[0.5],
    'iflag_ftype':3       # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}
Ftype2_para={
    'numOf3bfeat1':[6],
    'numOf3bfeat2':[6],
    'Rc':[6.1],
    'Rc2':[12.2],
    'Rm':[5.0],
    'iflag_grid':[3],                      # 1 or 2 or 3
    'fact_base':[0.2],
    'dR1':[0.5],
    'dR2':[0.5],
    'iflag_ftype':3   # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}
nFeatures=162
E_tolerance=0.3
# iflag_ftype=3        # Seems like, this should be in the Ftype1/2_para        # 2 or 3 or 4 when 4, iflag_grid must be 3
recalc_grid=1                      # 0:read from file or 1: recalc 
#----------------------------------------------------
rMin=0.0
#************** cluster input **********************
kernel_type=2             # 1 is exp(-(dd/width)**alpha), 2 is 1/(dd**alpha+k_dist0**alpha)
use_lpp=True
lppNewNum=3               # new feature num lpp generate. you can adjust more lpp parameters by editing feat_LPP.py. also see explains in it
lpp_n_neighbors=5         
lpp_weight='adjacency'    # 'adjacency' or 'heat'
lpp_weight_width=1.0
alpha0=1.0
k_dist0=0.01                       
DesignCenter=False
ClusterNum=[3,2]
#-----------------------------------------------

#******** for fit.input *******************************

fortranFitAtomRepulsingEnergies=[0.000]            #fortran fitting时对每种原子设置的排斥能量的大小，此值必须设置，无default值！(list_like)
fortranFitAtomRadii=[0]                        #fortran fitting时对每种原子设置的半径大小，此值必须设置，无default值！(list_like)
fortranFitWeightOfEnergy=6.2                    #fortran fitting时最后fit时各个原子能量所占的权重(linear和grr公用参数)  default:0.9
fortranFitWeightOfEtot=53.9                      #fortran fitting时最后fit时Image总能量所占的权重(linear和grr公用参数)  default:0.0
fortranFitWeightOfForce=4.31                     #fortran fitting时最后fit时各个原子所受力所占的权重(linear和grr公用参数)  default:0.1
fortranFitRidgePenaltyTerm=0.0001               #fortran fitting时最后岭回归时所加的对角penalty项的大小(linear和grr公用参数)  default:0.0001
fortranFitDwidth = 0.01
#----------------------------------------------------

#*********************** for MD **********************

#以下部分为md设置的参数 
mdCalcModel='lin'                               #运行md时，计算energy和force所用的fitting model，‘lin' or 'vv'
mdRunModel='nvt'                                #md运行时的模型,'nve' or 'nvt' or 'npt' or 'opt', default:'nve'
mdStepNum=10                                  #md运行的步数,default:1000
mdStepTime=1                                  #md运行时一步的时长(fs), default:1.0
mdStartTemperature=300                          #md运行时的初始温度
mdEndTemperature=300                            #md运行采用'nvt'模型时，稳定温度(or npt)
mdNvtTaut=0.1*1000                               #md运行采用'nvt'模型时，Berendsen温度对的时间常数 (or npt)
mdOptfmax=0.05
mdOptsteps=10

isTrajAppend=False                              #traj文件是否采用新文件还是接续上次的文件  default:False
isNewMovementAppend=False                       #md输出的movement文件是采用新文件还是接续上次的文件  default:False
mdTrajIntervalStepNum=50
mdLogIntervalStepNum=10
mdNewMovementIntervalStepNum=10
mdStartImageIndex=0                             #若初始image文件为MOVEMENT,初始的image的编号  default:0

isOnTheFlyMd=False                              #是否采用on-the-fly md,暂时还不起作用  default:False
isFixedMaxNeighborNumForMd=False                #是否使用固定的maxNeighborNum值，默认为default,若为True，应设置mdMaxNeighborNum的值
mdMaxNeighborNum=None                           #采用固定的maxNeighborNum值时，所应该采用的maxNeighborNum值(目前此功能不可用)

isMdCheckVar=False                               #若采用 'grr' model时，是否计算var  default:False
isReDistribute=True                             #md运行时是否重新分配初速度，目前只是重新分配, default:True
velocityDistributionModel='MaxwellBoltzmann'    #md运行时,重新分配初速度的方案,目前只有'MaxwellBoltzmann',default:MaxwellBoltzmann

isMdProfile=False

#-------------------------------------------------------
#********************* NN_related ***************
# device related

gpu_mem  = 0.9       # tensorflow used gpu memory
cuda_dev = '0'       # unoccupied gpu, using 'nvidia-smi' cmd
cupyFeat=True
tf_dtype = 'float32' # dtype of tensorflow trainning, 'float32' faster than 'float64'
torch_dtype = 'float32'
#================================================================================
# NN model related
activation_func='softplus'     # could choose 'softplus' and 'elup1' now
ntypes=len(atomType)
nLayers = 3
nNodes = np.array([[64,64],[32,32],[1,1]])
b_init=np.array([16.2,17.5])      # energy of one atom, for different types, just a rough value
#nLayers = 4
#nNodes = np.array([[16,],[64,],[32,],[1,]])
dwidth = 0.01

#================================================================================
# trainning 
train_continue = False     #是否接着训练
progressbar = False 
flag_plt = False
train_stage = 1      # only 1 or 2, 1 is begining training from energy and then force+energy, 2 is directly training from force+energy
train_verb = 0       

learning_rate= 1e-3
batch_size = 40        
rtLossE      = 6.96     # weight for energy, NN fitting 各个原子能量所占的权重
rtLossF      = 4.97     # weight for force, NN fitting 各个原子所受力所占的权重
rtLossEtot      = 59.2     # weight for Etot, NN fitting 各个原子所受力所占的权重
bias_corr = True
#epochs_pretrain = 1001
epochs_alltrain = 10001     # energy 训练循环次数
epochs_Fi_train = 2001       # force+energy 训练循环次数 

iFi_repeat      = 1
eMAE_err = 0.01 # eV
fMAE_err = 0.02 # eV/Ang


#************* no need to edit ****************************
#fortranFitAtomTypeNum=0                        #fortran fitting时原子所属种类数目(linear和grr公用参数)  default:0(废弃，不需要)
# fortranFitFeatNum0=None                         #fortran fitting时输入的feat的数目(linear和grr公用参数)  default:None
# fortranFitFeatNum2=None                         #fortran fitting时PCA之后使用的feat的数目(linear和grr公用参数)  此值目前已经不需要设置
isDynamicFortranFitRidgePenaltyTerm=False       #fortran fitting时最后岭回归时所加的对角penalty项的大小是否根据PCA最小的奇异值调整 default:False
fortranGrrRefNum=[800,1000]                           #fortran grr fitting时每种原子所采用的ref points数目,若设置应为类数组   default:None
fortranGrrRefNumRate=0.1                        #fortran grr fitting时每种原子选择ref points数目所占总case数目的比率   default:0.1
fortranGrrRefMinNum=1000                        #fortran grr fitting时每种原子选择ref points数目的下限数目，若case数低于此数，则为case数
fortranGrrRefMaxNum=3000                        #fortran grr fitting时每种原子选择ref points数目的上限数目，若设定为None，则无上限(不建议)
fortranGrrKernelAlpha=1                         #fortran grr fitting时kernel所用超参数alpha
fortranGrrKernalDist0=3.0                       #fortran grr fitting时kernel所用超参数dist0
realFeatNum=111

#-----------------------------------------------


trainSetDir=os.path.abspath(trainSetDir)
genFeatDir=os.path.abspath(genFeatDir)
fortranFitSourceDir=os.path.abspath(fortranFitSourceDir)
fbinListPath=os.path.join(trainSetDir,'location')
sourceFileList=[]
InputPath=os.path.abspath('./input/')
OutputPath=os.path.abspath('./output/')
Ftype1InputPath=os.path.join('./input/',Ftype_name[1]+'.in')
Ftype2InputPath=os.path.join('./input/',Ftype_name[2]+'.in')
featCollectInPath=os.path.join(fitModelDir,'feat_collect.in')
fitInputPath_lin=os.path.join(fitModelDir,'fit_linearMM.input')
fitInputPath2_lin=os.path.join(InputPath,'fit_linearMM.input')
featCollectInPath2=os.path.join(InputPath,'feat_collect.in')
# featCalcInfoPath=os.path.join(trainSetDir,'feat_calc_info.txt')

# featTrainTxt=os.path.join(trainSetDir,'trainData.txt')
# featTestTxt=os.path.join(trainSetDir,'testData.txt')

if fitModelDir is None:
    fitModelDir=os.path.join(fortranFitSourceDir,'fread_dfeat')
else:
    fitModelDir=os.path.abspath(fitModelDir)
linModelCalcInfoPath=os.path.join(fitModelDir,'linear_feat_calc_info.txt')
# grrModelCalcInfoPath=os.path.join(fitModelDir,'gaussion_feat_calc_info.txt')
# fitInputPath=os.path.join(fitModelDir,'fit.input')
linFitInputBakPath=os.path.join(fitModelDir,'linear_fit_input.txt')
# grrFitInputBakPath=os.path.join(fitModelDir,'gaussion_fit_input.txt')

f_atoms=os.path.join(mdImageFileDir,'atom.config')
atomTypeNum=len(atomType)
# if os.path.exists(fitInputPath2):
#     with open(fitInputPath2,'r') as sourceFile:
#         sourceFile.readline()
#         line=sourceFile.readline()
#         if len(line) > 1 :
#             realFeatNum=int(line.split(',')[1])
#         else:
#             pass
nFeats=np.array([realFeatNum,realFeatNum,realFeatNum])
# dir_work = os.path.join(trainSetDir,'NN_output/')          # The major dir that store I/O files and data
dir_work = os.path.join(fitModelDir,'NN_output/')
# f_post  = '.csv'              # postfix of feature files
# f_txt_post = '.txt'

# dir_feat = dir_work + "features/"                              
# f_pretr_feat = dir_feat+f_feat +"_feat_pretrain"+f_post
f_train_feat = os.path.join(dir_work,'feat_train.csv')
f_test_feat = os.path.join(dir_work,'feat_test.csv')
# f_test_feat  = dir_feat+f_feat +"_feat_test"+f_post
# f_pretr_natoms = dir_feat+f_feat+"_nat_pretrain"+f_post
f_train_natoms = os.path.join(dir_work,'natoms_train.csv')
f_test_natoms = os.path.join(dir_work,'natoms_test.csv')                                 
# f_pretr_feat = dir_feat+f_feat +"_feat_pretrain"+f_post
# f_train_feat = os.path.join(dir_work,'dE_file_train.csv')
# f_test_feat  = os.path.join(dir_work,'dE_file_train.csv')
# f_pretr_dfeat = dir_feat+f_feat +"_d_pretrain"+f_txt_post
f_train_dfeat = os.path.join(dir_work,'dfeatname_train.csv')
f_test_dfeat  = os.path.join(dir_work,'dfeatname_test.csv')
f_train_egroup = os.path.join(dir_work,'egroup_train.csv')
f_test_egroup  = os.path.join(dir_work,'egroup_test.csv')

# f_pretr_nblt = dir_feat+f_feat +"_nblt_pretrain"+f_post
# f_train_nblt = dir_feat+f_feat +"_nblt_train"+f_post
# f_test_nblt  = dir_feat+f_feat +"_nblt_test"+f_post
# dfeat_dir = dir_feat+f_feat + '_dfeat/'

d_nnEi  = os.path.join(dir_work,'NNEi/')
d_nnFi  = os.path.join(dir_work,'NNFi/')
f_Einn_model   = d_nnEi+'allEi_final.ckpt'
f_Finn_model   = d_nnFi+'Fi_final.ckpt'
f_data_scaler = d_nnFi+'data_scaler.npy'
f_Wij_np  = d_nnFi+'Wij.npy'
