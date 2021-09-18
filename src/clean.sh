#!/bin/bash

cd pre_data/gen_feature
make clean
cd ../fit
make clean
cd ../fortran_code
make clean
cd ../../test/MD
make clean

