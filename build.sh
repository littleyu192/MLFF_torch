#!/bin/bash

mkdir bin
cd data/gen_feature
make
cd ../fit
make
cd ../..