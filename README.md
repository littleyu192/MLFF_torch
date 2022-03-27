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
	# generate features
	cd the/path/to/data    # in parameter.py, make sure isCalcFeat=True && isFitVdw=False
	ulimit -Ss unlimited
	python the/path/to/MLFF_torch/src/bin/mlff.py
	python the/path/to/MLFF_torch/src/bin/seper.py  # in parameters.py, test_ratio = 0.2 for default
	python the/path/to/MLFF_torch/src/bin/gen_data.py
	# model train, mannul in MLFF_torch/dataset/cu1000/parameters*.py
	# if you want use deepmd model in training
	cp parameters_dp.py parameters.py
	# if u have muti-MOVEMENT file in PWdata directory, in parameters.py, make sure batch_size = 1
	python the/path/to/MLFF_torch/src/train.py --deepmd=True -n DeepMD_cfg_dp -s record
	# if you want use MLP by kalmane filter
	cp parameters_kf.py parameters.py
	or python the/path/to/MLFF_torch/src/train.py -s record 
	# model test
```

### Code contribution guide
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
