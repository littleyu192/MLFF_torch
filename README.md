# MLFF

> MLFF is a machine learning(ML) based method of force fields (FF) for molecular dynamics simulations, which is written in Python/Fortran.  This project uses 8 different types of features (typically translational, rotation, and permutation invariant), aims at retaining the accuracy of ab initio density functional theory (DFT) . 
>
>  More specifically, Type1: 2B features(use piecewise cosine basis functions); Type2: 3B features(three sub-types, just like in 2b); Type3: 2B Gaussian (use Gaussian function multiplying a mask function to make it smooth); Type4: 3Bcos(based on the idea of bond angles); Type5: Multiple Tensor Potential(uses the amplitude of charge, dipole, quadrapole, etc); Type6: SNAP (spectral neighbor analysis potential); Type7: deepMD1; Type8: deepMD2.

## Getting Started 

项目使用条件、如何安装部署、怎样运行使用以及使用演示

### Prerequisites  and  Installation

export MKLROOT=/the/path/to/mkl

```
with conda:
	# create conda env
	conda create -n *name* python=3.8
	conda activate *name*
	conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
	conda install -c conda-forge cupy
	# compiler
	cd src
	./build.sh
    # export MLFF bin to bashrc
    vim ~/.bashrc
    export PATH=the/path/to/MLFF_torch/src/bin:$PATH
    source ~/.bashrc
    conda deactivate *name*
    
    
```

```sh
with dorcker:

```

### Usage example 

	# generate features
	cd the/path/to/data    # in parameter.py, make sure isCalcFeat=True && isFitLinModel=True
	python the/path/to/MLFF_torch/src/bin/mlff.py
	python the/path/to/MLFF_torch/src/bin/seper.py  # in parameter.py, test_ratio = 0.2 for default
	python the/path/to/MLFF_torch/src/bin/seper.py 
	# model train and test
	python the/path/to/MLFF_torch/src/train.py
	python the/path/to/MLFF_torch/src/test.py
## License 

If you use this code in any future publications, please cite this: