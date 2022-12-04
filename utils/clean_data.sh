#! /bin/bash

rm -rf fread_dfeat input output __pycache__ train_data test
cd PWdata
rm `find . | xargs ls -ld | grep -v "MOV" | grep ^- | awk '{print $9}'`
rm MOVEMENTall
cd ..
