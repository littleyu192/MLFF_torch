#!/bin/bash

mkdir bin
mkdir lib
cd data/gen_feature
make
cd ../fit
make
cd ../..
cd src/fortran_code
make
cd ../..