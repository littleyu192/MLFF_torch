# MLFF

MLFF is a machine learning(ML) based method of force fields (FF) for molecular dynamics simulations, which is written in Python/Fortran.
1. This project contains 6 different types of features. More specifically, Type1: [2B features](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.064103) use piecewise cosine basis functions; Type2: [3B features](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.064103) three sub-types, just like in 2b; Type3: [2B Gaussian](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401) use Gaussian function multiplying a mask function to make it smooth; Type4: [3Bcos](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401) based on the idea of bond angles; Type5: [Multiple Tensor Potential](https://iopscience.iop.org/journal/2632-2153) uses the amplitude of charge, dipole, quadrapole, etc; Type6: [SNAP](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.184115) [spectral neighbor analysis potential](https://www.sciencedirect.com/science/article/pii/S0021999114008353); The features have the following properties: translational, rotation, and permutation invariant.

2. This project contains two kinds of optimizer, [ADAM](https://dblp.org/rec/journals/corr/KingmaB14.html) and [EKF](https://onlinelibrary.wiley.com/doi/book/10.1002/0471221546). 

We implemented GKF based MLP(users can choose the above features), [Adam based DeePMD](https://proceedings.neurips.cc/paper/2018/hash/e2ad76f2326fbc6b56a45a56c59fafdb-Abstract.html) and LKF based DeePMD. 



## Table of Contents

- [ChangeLog](#ChangeLog)
- [Installation](#Installation)
- [Quick start](#Quickstart)
- [Usage](#usage)
	- [MLP](#MLP)
	- [DP](#DP)
	- [DPKF](#DPKF)
- [ContributionGuidance](#ContributionGuidance)
- [License](#license)

## ChangeLog
- framework reorganiztion
- support P matrix saving in checkpoint
- support tens of thousands of images generating features (DP, DPKF)
- splitting dataset in small sets


## Installation

export MKLROOT=/the/path/to/mkl

```sh
with conda:
	# create conda env
	conda create -n *name* python=3.8
	conda activate *name*
	conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
	conda install -c pytorch magma-cuda110  # use the magma-cuda* that matches your CUDA version
	git clone --recursive https://github.com/pytorch/pytorch
	cd pytorch
	# if you are updating an existing checkout
	git submodule sync
	git submodule update --init --recursive --jobs 0
	export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
	python setup.py install
```

```sh
with dorcker:

```

```sh
horovod installation(optional):
	conda activate *name*
	# download NCCL  in https://developer.nvidia.com/nccl
	HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_INCLUDE=/home/husiyu/tools/nccl_2.15/include HOROVOD_NCCL_LIB=/home/husiyu/tools/nccl_2.15/lib  pip install horovod

```

```sh
	# compile
	cd /the/path/the/MLFF_torch/src
	./build.sh
	export PATH=the/path/to/MLFF_torch/src/bin:$PATH
	cd op && python setup.py install
	conda deactivate *name*
```

## Quick start
1. **template** cu_parameters_dpkf_template.py, cu_parameters_nnkf_template.py
2. in src/pre_data, **data processing** e.g., feature generating
3. in src/loss, **customized loss function**, e.g., dpmd loss calculation
4. in src/model, **MODEL!!!**, e.g., MLFF, dp
5. in src/optimizer, **OPTIMIZER!!!**, e.g., LKF, GKF
6. in src/op, **calculate force**, the customized c++ operartor
7. in src/trainer.py, the defined train, train_kf, valid functions
8. in src/train_hvd.py, the horovod version of distributed training version
9. in src/main.py, the training process must be called python file


## Usage

### MLP(NNKF)
> generate features
```sh
	cd the/path/to/data    # in parameter.py, make sure isCalcFeat=True && isFitVdw=False
	ulimit -Ss unlimited
	# cu_parameters_nnkf_template.py is a template of Cu system, nnkf method
	cp the/path/to/MLFF_torch/cu_parameters_nnkf_template.py parameters.py
	python the/path/to/MLFF_torch/src/pre_data/mlff.py
	python the/path/to/MLFF_torch/src/pre_data/seper.py  # in parameters.py, test_ratio = 0.2 for default
	python the/path/to/MLFF_torch/src/pre_data/gen_data.py 
```
> model training
1. Train in one GPU
```sh
	python the/path/to/MLFF_torch/src/main.py --gpu 0 -b 1 --opt GKF --epochs 30 -s GKFrecord
	# --gpu 0(0 is the idx of GPU)
	# -b 1(the batch size is set 1 for training)
	# --opt GKF(use the default optimizer if using MLP net)
	# -s GKFrecord(assign the directory of stored model and log file)
```

2. Train in multi GPUs(One node)
```sh
	python the/path/to/MLFF_torch/src/main.py --dist-url 'tcp://127.0.0.1:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -b 4 --opt GKF --epochs 30 -s GKF_4gpus
	# do not need to assigh GPU 
	# -b 4(the batch size should set an integer multiple of the used GPUs, e.g.,4, 8, 12)
	# --opt GKF(use the default optimizer if using MLP net)
	# -s GKFrecord(assign the directory of stored model and log file)
```

3. Train in multi GPUs(Multi nodes)
```sh
	# in the root node:
	python the/path/to/MLFF_torch/src/main.py --dist-url 'tcp://127.0.0.1:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 -b 8 --opt GKF --epochs 30 -s GKF_8gpus
	# in the child nodes:
	python the/path/to/MLFF_torch/src/main.py --dist-url 'tcp://$root_node_IP$:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 -b 8 --opt GKF --epochs 30 -s GKF_8gpus
	# do not need to assigh GPU 
	# -b 8(the batch size should set an integer multiple of the used GPUs, e.g.,8, 16, etc)
	# --opt GKF(use the default optimizer if using MLP net)
	# -s GKFrecord(assign the directory of stored model and log file)
```

### DP
> generate features
```sh
	cd the/path/to/data
	ulimit -Ss unlimited
	# cu_parameters_dpkf_template.py is a template of Cu system, dp method
	cp the/path/to/MLFF_torch/src/cu_config_template.yaml config.yaml
	python the/path/to/MLFF_torch/src/pre_data/dp_mlff.py
```
> model training
1. Train in one GPU
```sh
	python the/path/to/MLFF_torch/src/dp_main.py --gpu 0 -b 1 --opt ADAM --epochs 1000 -s dprecord
	(or: CUDA_VISIBLE_DEVICES=0 python the/path/to/MLFF_torch/src/dp_train_hvd.py --opt ADAM --epochs 30 -s dprecord)
	# --gpu 0(0 is the idx of GPU)
	# -b 1(the batch size is set 1 for training)
	# --opt ADAM(use the default optimizer if using DP net)
	# -s dprecord(assign the directory of stored model and log file)
```

2. Train in multi GPUs(One node)
```sh
	python the/path/to/MLFF_torch/src/dp_main.py --dist-url 'tcp://127.0.0.1:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -b 4 --opt ADAM --epochs 1000 -s dp_4gpus
	(or:  horovodrun -np 4 python the/path/to/MLFF_torch/src/dp_train_hvd.py --opt ADAM -b 8 --epochs 200 --hvd -s dp_4gpus)
	# do not need to assigh GPU 
	# -b 4(the batch size should set an integer multiple of the used GPUs, e.g.,4, 8, 12)
	# --opt ADAM(use the default optimizer if using DP net)
	# -s dp_4gpus(assign the directory of stored model and log file)
```

3. Train in multi GPUs(Multi nodes)
```sh
	# in the root node:
	python the/path/to/MLFF_torch/src/dp_main.py --dist-url 'tcp://127.0.0.1:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 -b 8 --opt ADAM --epochs 1000 -s dp_8gpus
	# in the child nodes:
	python the/path/to//MLFF_torch/src/main.py --dist-url 'tcp://$root_node_IP$:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 -b 8 --opt ADAM --epochs 1000 -s dp_8gpus
	# do not need to assigh GPU 
	# -b 8(the batch size should set an integer multiple of the used GPUs, e.g.,8, 16, etc)
	# --opt ADAM(use the default optimizer if using DP net)
	# -s dp_8gpus(assign the directory of stored model and log file)
	(or: horovodrun -np 8 -H server1:2,server2:2 python the/path/to/MLFF_torch/src/dp_train_hvd.py --opt ADAM -b 8 --epochs 200 --hvd -s dp_8gpus)
```


### DPKF
- generate features part: the same as DP
- model training part: change --opt ADAM to --opt LKF

> model validation
```sh
	python the/path/to/MLFF_torch/src/dp_main.py --opt LKF -b 1 -s dpkfrecord -r -e

	python the/path/to/MLFF_torch/src/dp_main.py --dp --opt LKF --dist-url 'tcp://127.0.0.1:1235' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -b 32 -s dpkf_4gpus -r -e
	# -s follows the directory of model you wanna evaluate
	# -r means recover
	# -e means evaluate
```

## ContributionGuidance
```sh
	git checkout master
	git pull origin master
	git checkout -b your_branch_name  # e.g., hsy_dev
	# start your code in your branch
	# if your're ready to pull request
	git checkout master
	git pull origin master
	git checkout your_branch_name
	git rebase origin/master
	git push origin your_branch_name
	# in github, click pull request
```

## License

If you use this code in any future publications, please cite this:



**If you have any questions, contact husiyu20b@ict.ac.cn**

