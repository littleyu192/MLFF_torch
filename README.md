# MLFF

> MLFF is a machine learning(ML) based method of force fields (FF) for molecular dynamics simulations, which is written in Python/Fortran.  This project uses 8 different types of features (typically translational, rotation, and permutation invariant), aims at retaining the accuracy of ab initio density functional theory (DFT) . 
>
>  More specifically, Type1: 2B features(use piecewise cosine basis functions); Type2: 3B features(three sub-types, just like in 2b); Type3: 2B Gaussian (use Gaussian function multiplying a mask function to make it smooth); Type4: 3Bcos(based on the idea of bond angles); Type5: Multiple Tensor Potential(uses the amplitude of charge, dipole, quadrapole, etc); Type6: SNAP (spectral neighbor analysis potential); Type7: deepMD1; Type8: deepMD2.

## Getting Started

### Prerequisites  and  Installation

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

	# compiler
	cd /the/path/the/MLFF_torch/src
	./build.sh
    export PATH=the/path/to/MLFF_torch/src/bin:$PATH
	cd op && python setup.py install
    conda deactivate *name*
    
    
```

```sh
with dorcker:

```

### Usage example 
```sh
	# NN and NNKF generate features
	cd the/path/to/data    # in parameter.py, make sure isCalcFeat=True && isFitVdw=False
	ulimit -Ss unlimited
    # cu_parameters_nnkf_template.py is a template of Cu system, nnkf method
	cp the/path/to/MLFF_torch/cu_parameters_nnkf_template.py parameters.py
    python the/path/to/MLFF_torch/src/bin/mlff.py
	python the/path/to/MLFF_torch/src/bin/seper.py  # in parameters.py, test_ratio = 0.2 for default
	python the/path/to/MLFF_torch/src/bin/gen_data.py
    # training, make sure in parameters.py use_GKalman = 1 
	python the/path/to/MLFF_torch/src/train.py -s nnkf_record
	# if you want use NN without kalman filter, switch use_GKalman = 0 in parameters.py 
	python the/path/to/MLFF_torch/src/train.py -s nn_record
	
	# DP and DPKF generate features
	cd the/path/to/data    # in parameter.py, make sure isCalcFeat=True && isFitVdw=False
	ulimit -Ss unlimited
    # cu_parameters_dpkf_template.py is a template of Cu system, dpkf method
	cp the/path/to/MLFF_torch/cu_parameters_dpkf_template.py parameters.py
    python the/path/to/MLFF_torch/src/bin/mlff.py
	python the/path/to/MLFF_torch/src/bin/seper.py  # in parameters.py, test_ratio = 0.2 for default
	python the/path/to/MLFF_torch/src/bin/gen_dpdata.py
    # if you want use deepmd model in training, make sure dR_neigh=True && use_Ftype =[1]
	# if u have muti-MOVEMENT file in PWdata directory, in parameters.py, make sure batch_size = 1
	python the/path/to/MLFF_torch/src/train.py --dp=True -n DeepMD_cfg_dp_kf --nselect=48 --blocksize=10240 --groupsize=6 -s dpkf_record
	# the recomended nselect is 24, 48, 72; the recommended blocksize is 5120, 10240; the recommended groupsize is 6, 12 
    # if you want use DP without kalman filter, switch use_L1Kalman = 0 in cu_parameters_dpkf_template.py 
	python the/path/to/MLFF_torch/src/train.py --dp=True -n DeepMD_cfg_dp -s dp_record
	# model test
```

### Code contribution guidance
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
