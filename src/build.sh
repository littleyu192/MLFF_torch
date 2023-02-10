#!/bin/bash

mkdir bin
mkdir lib
make -C pre_data/gen_feature
make -C pre_data/fit
make -C pre_data/fortran_code  # spack load gcc@7.5.0
#make -C test/MD
#cd QCAD/fortran_code
#make libiotk.a
#cd ../..
make -C md/fortran_code
cd bin
ln -s ../clean_data.sh .
chmod +x clean_data.sh
#ln -s ../pre_data/mlff.py .
#ln -s ../pre_data/seper.py .
#ln -s ../pre_data/gen_data.py .
#ln -s ../pre_data/data_loader_2type.py .
#ln -s ../train.py .
#ln -s ../test.py .
#ln -s ../predict.py . 

#chmod +x ./mlff.py
#chmod +x ./seper.py
#chmod +x ./gen_data.py
#chmod +x ./data_loader_2type.py
#chmod +x ./train.py
#chmod +x ./test.py
#chmod +x ./predict.py
cd ..            # back to src dir
