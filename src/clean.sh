#!/bin/bash
rm -f bin/*.r
rm -f bin/*.x
make clean -C pre_data/gen_feature
make clean -C pre_data/fit
make clean -C pre_data/fortran_code
#make clean -C QCAD/fortran_code
#make clean -C test/MD

