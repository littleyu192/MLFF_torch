import numpy as np
import os
import torch
import torch.nn.functional as F

isCalcFeat=1
isFitLinModel=0
isClassify=False
isRunMd=False                                   #是否训练运行md  default:False
isRunMd_nn=False
isFollowMd=False                                #是否是接续上次的md继续运行  default:False
isFitVdw=False  #训练时需关掉
isRunMd100_nn=False
isRunMd100=False
add_force=False     # for NN md
pbc = False
is_nn_do_profile = False
#************** Dir **********************

prefix = r'./'
trainSetDir = r'./PWdata'
codedir=r'/data/data/husiyu/software/MLFF_torch'
fortranFitSourceDir=codedir+'/src/pre_data/fit/'
fitModelDir = r'./fread_dfeat'
train_data_path = r'./train_data/final_train'
test_data_path = r'./train_data/final_test'
dRneigh_path = trainSetDir + r'/dRneigh.dat'
test_ratio = 0.2

genFeatDir = r'./gen_feature'
mdImageFileDir=r'./MD'                              #设置md的初始image的文件所在的文件夹  default:'.'

#********* for gen_feature.in *********************
atomType=[29]                                  #铜有29个同位素,相当于29个种类的cu
maxNeighborNum=100
natoms=[108]

iflag_PCA=0
Rc_M=6.0                     # max of Rcut

Ftype_name={1:'gen_2b_feature', 2:'gen_3b_feature',
            3:'gen_2bgauss_feature', 4:'gen_3bcos_feature',
            5:'gen_MTP_feature', 6:'gen_SNAP_feature',
            7:'gen_deepMD1_feature', 8:'gen_deepMD2_feature',
            }

#use_Ftype=[1,2,3,4,5,6,7,8]
use_Ftype=[1, 2]
nFeatures=42
dR_neigh = 0

is_scale = False
itype_Ei_mean=[166.46]
use_storage_scaler = False
storage_scaler = False

n_epoch=30

DeepMD_cfg = {
    'embeding_net': {
        'network_size': [16, 32, 64], # 第一维表示输入的维度
	'bias': True,
	'resnet_dt': True,
	'activation': F.softplus,    #torch.sigmoid,
	},
    'fitting_net': {
	'network_size': [120, 120, 120, 1],
	'activation': F.softplus,    #torch.sigmoid,
	'bias': True,
	}
}

DeepMD_cfg_dp = {
    'embeding_net': {
        'network_size': [25, 50, 100], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [240, 240, 240, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}

DeepMD_cfg_dp_kf = {
    'embeding_net': {
        'network_size': [25, 25, 25], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [50, 50, 50, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}
DeepMD_cfg_dp_kf_mini = {
    'embeding_net': {
        'network_size': [5, 5, 5], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [10, 10, 10, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}

MLFF_dmirror_cfg1 = [
	('linear', nFeatures, 30, True),
	('activation',),
	('linear', 30, 60, True),
	('activation',),
	('linear', 60, 1, True)
	]

nfeat_type=len(use_Ftype)
Ftype1_para={
    'numOf2bfeat':[24,24],       # [itpye1,itype2]
    'Rc':[6.0,6.0],
    'Rm':[5.8,5.8],   # 'Rc':[5.5,5.5], 'Rm':[5.0,5.0],
    'iflag_grid':[3,3],                      # 1 or 2 or 3
    'fact_base':[0.2,0.2],
    'dR1':[0.5,0.5],
    'iflag_ftype':3       # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}
Ftype2_para={
    'numOf3bfeat1':[3,3],     # 3*3=9
    'numOf3bfeat2':[3,3],     # 3*3=9   总的特征数24+9+9=42
    'Rc':[5.5,5.5],
    'Rc2':[5.5,5.5],
    'Rm':[5.0,5.0],
    'iflag_grid':[3,3],                      # 1 or 2 or 3
    'fact_base':[0.2,0.2],
    'dR1':[0.5,0.5],
    'dR2':[0.5,0.5],
    'iflag_ftype':3   # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}
Ftype3_para={           # 2bgauss
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'n2b':[6 for tmp in range(10)],       # number of elements in n2b = num atom type
    'r':[ [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0,
           3.5, 3.5, 3.5, 5.0, 5.0, 5.0, 4.5, 4.5, 4.5, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5,
        ] for tmp in range(10) ],
    'w':[ [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
           1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
          ] for tmp in range(10) ]
}
Ftype4_para={           # 3bcos
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'n3b':[20 for tmp in range(10)],       # number of elements in n2b = num atom type
    'eta':   [ [1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0,
                1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 
                1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 
                1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 1.0/1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 
             ] for tmp in range(10)],
    'w':     [ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
             ] for tmp in range(10)],
    # w is the \ksi in formula
    'lambda':[ [1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0,
                1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0,
                1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0,
               ] for tmp in range(10)],
}
Ftype5_para={           # MTP
    'Rc':[6.0 for tmp in range(10)],     # number of elements in Rc = num atom type
    'Rm':[0.01  for tmp in range(10)],     # number of elements in Rc = num atom type
    'n_MTP_line': [5 for tmp in range(10)], # 5~11
    'tensors':[
                [
                  '1, 4, 0, ( )                              ',
                  '2, 3,3, 0,0, ( ), ( )                     ',
                  '2, 3,3, 1,1, ( 21 ), ( 11 )               ',
                  '2, 3,3, 2,2, ( 21, 22 ), ( 11, 12 )       ',
                  '3, 2,2,2, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )',
                  '3, 2,2,2, 3,2,1 ( 21, 22, 31 ), ( 11, 12 ), ( 13 )',
                  '3, 2,2,2, 4,2,2 ( 21, 22, 31, 32 ), ( 11, 12 ), ( 13, 14 )',
                  '4, 2,2,2,2 3,1,1,1 ( 21, 31, 41 ), ( 11 ), ( 12 ), ( 13 )',
                  '4, 2,2,2,2 4,2,1,1 ( 21, 22, 31, 41 ), ( 11, 12 ), ( 13 ), ( 14 )',
                  '4, 2,2,2,2 5,2,2,1 ( 21, 22, 31, 32, 41 ), ( 11, 12 ), ( 13, 14 ), ( 15 )',
                ] for tmp in range(10)
              ],
    }
Ftype6_para={
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'J' :[3.0 for tmp in range(10)],
    'n_w_line': [2 for tmp in range(10)],
    'w1':[ [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]  for tmp in range(10)],  # shape(w1) = (ntype, n_w_line)
    'w2':[ [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]  for tmp in range(10) ],
    }
Ftype7_para={
    'Rc':[5.4  for tmp in range(10)],     # number of elements in Rc = num atom type
    'Rc2':[3.0  for tmp in range(10)],
    'Rm':[1.0  for tmp in range(10)],
    'M': [4  for tmp in range(10)],
    'weight_r': [1.0  for tmp in range(10)],
    }
Ftype8_para={
    'Rc':[5.4  for tmp in range(10)],     # number of elements in Rc = num atom type
    'M':[8  for tmp in range(10)],
    'weight_r':[1.0  for tmp in range(10)],
    'rg':[
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0]  for tmp in range(10)
         ],
    'w':[
            [1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5]  for tmp in range(10)
        ]
    }



E_tolerance=999.0
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

fortranFitAtomRepulsingEnergies=[0.000,0.000]            #fortran fitting时对每种原子设置的排斥能量的大小，此值必须设置，无default值！(list_like)
fortranFitAtomRadii=[2.83]                        #fortran fitting时对每种原子设置的半径大小，此值必须设置，无default值！(list_like)
fortranFitWeightOfEnergy=0.8                    #fortran fitting时最后fit时各个原子能量所占的权重(linear和grr公用参数)  default:0.9
fortranFitWeightOfEtot=0.0                      #fortran fitting时最后fit时Image总能量所占的权重(linear和grr公用参数)  default:0.0
fortranFitWeightOfForce=0.2                     #fortran fitting时最后fit时各个原子所受力所占的权重(linear和grr公用参数)  default:0.1
fortranFitRidgePenaltyTerm=0.0001               #fortran fitting时最后岭回归时所加的对角penalty项的大小(linear和grr公用参数)  default:0.0001
fortranFitDwidth=3.0
dwidth=3.0
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

feature_dtype = 'float64'
training_dtype = 'float64'
inference_dtype = 'float64'

# device related

gpu_mem  = 0.9       # tensorflow used gpu memory
cuda_dev = '0'       # unoccupied gpu, using 'nvidia-smi' cmd
cupyFeat=True
torch_dtype = 'float64'
tf_dtype = 'float64' # dtype of tensorflow trainning, 'float32' faster than 'float64'
#================================================================================
# NN model related
activation_func='softplus'     # could choose 'softplus' and 'elup1' now
ntypes=len(atomType)
nLayers = 3
nNodes = np.array([[15,15],[15,15],[1,1]])
#nLayers=3
#nNodes = np.array([[120,120],[120,120],[120,120],[1,1]])
b_init=np.array([166.3969])      # energy of one atom, for different types, just a rough value
DCNLayers = 5

#================================================================================
# training 
train_continue = False     #是否接着训练
progressbar = False 
flag_plt = False
train_stage = 2      # only 1 or 2, 1 is begining training from energy and then force+energy, 2 is directly training from force+energy
train_verb = 0       
learning_rate= 1e-3
batch_size = 1
#rtLossE      = 0.6     # weight for energy, NN fitting 各个原子能量所占的权重
#rtLossF      = 0.2     # weight for force, NN fitting 各个原子所受力所占的权重
#rtLossEtot   = 0.2
rtLossE = 0.8
rtLossF = 0.2
rtLossEtot = 0

bias_corr = True
epochs_alltrain = 6000     # energy 训练循环次数
epochs_Fi_train = 1000       # force+energy 训练循环次数 1000个epoch效果较好

iFi_repeat      = 1
eMAE_err = 0.01 # eV
fMAE_err = 0.02 # eV/Ang


#************* no need to edit ****************************
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
FtypeiiInputPath={i:'' for i in range(1,9)}  # python-dictionary, i = 1,2,3,4,5,6,7,8
for i in range(1,9):
    FtypeiiInputPath[i]=os.path.join('./input/',Ftype_name[i]+'.in')
featCollectInPath=os.path.join(fitModelDir,'feat_collect.in')
fitInputPath_lin=os.path.join(fitModelDir,'fit_linearMM.input')
fitInputPath2_lin=os.path.join(InputPath,'fit_linearMM.input')
featCollectInPath2=os.path.join(InputPath,'feat_collect.in')

if fitModelDir is None:
    fitModelDir=os.path.join(fortranFitSourceDir,'fread_dfeat')
else:
    fitModelDir=os.path.abspath(fitModelDir)

print(fortranFitSourceDir)

linModelCalcInfoPath=os.path.join(fitModelDir,'linear_feat_calc_info.txt')
linFitInputBakPath=os.path.join(fitModelDir,'linear_fit_input.txt')

f_atoms=os.path.join(mdImageFileDir,'atom.config')
atomTypeNum=len(atomType)
nFeats=np.array([realFeatNum,realFeatNum,realFeatNum])
dir_work = os.path.join(fitModelDir,'NN_output/')
f_train_feat = os.path.join(dir_work,'feat_train.csv')
f_test_feat = os.path.join(dir_work,'feat_test.csv')
f_train_natoms = os.path.join(dir_work,'natoms_train.csv')
f_test_natoms = os.path.join(dir_work,'natoms_test.csv')                                 
f_train_dfeat = os.path.join(dir_work,'dfeatname_train.csv')
f_test_dfeat  = os.path.join(dir_work,'dfeatname_test.csv')

f_train_dR_neigh = os.path.join(dir_work,'dR_neigh_train.csv')
f_test_dR_neigh  = os.path.join(dir_work,'dR_neigh_test.csv')

f_train_force = os.path.join(dir_work,'force_train.csv')
f_test_force  = os.path.join(dir_work,'force_test.csv')

f_train_egroup = os.path.join(dir_work,'egroup_train.csv')
f_test_egroup  = os.path.join(dir_work,'egroup_test.csv')

f_train_ep = os.path.join(dir_work,'ep_train.csv')
f_test_ep  = os.path.join(dir_work,'ep_test.csv')

d_nnEi  = os.path.join(dir_work,'NNEi/')
d_nnFi  = os.path.join(dir_work,'NNFi/')
f_Einn_model   = d_nnEi+'allEi_final.ckpt'
f_Finn_model   = d_nnFi+'Fi_final.ckpt'
f_data_scaler = d_nnFi+'data_scaler.npy'
f_Wij_np  = d_nnFi+'Wij.npy'

#f_wij_txt = os.path.join(fitModelDir, "Wij.txt")
