#!/bin/bash

mkdir bin
mkdir lib
cd pre_data/gen_feature
make
cd ../fit
make
cd ../fortran_code
make
cd ../../bin
ln -s ../pre_data/mlff.py .
ln -s ../pre_data/seper.py .
ln -s ../pre_data/gen_data.py .
ln -s ../pre_data/data_loader_2type.py .
chmod +x ./mlff.py
chmod +x ./seper.py
chmod +x ./gen_data.py
chmod +x ./data_loader_2type.py
cd ..
cd test/MD
make
cd ../..

