#!/usr/bin/env python3
#-*- coding utf-8 -*-

import os
import sys
import random
fp='fit.input'
with open(fp,'r') as f:
    line=f.readline()
    nType=int(line.split(',')[0])
    nFeat2=[]
    for i in range(nType):
        line=f.readline()
        #nFeat2.append(int(line.split(',')[2].split()[0]))
        strLines=str(i+1)+' \n'+str(random.randint(-2100000000,-1))+' \n'
        command='echo "'+strLines+'" | ../feat_PCA_metric2.r'
        #print(strLines)
        #print(command)
        os.system(command)

'''
iType=int(sys.argv[1])
print(iType)
print(nFeat2[iType-1])
print(random.randint(-2100000000,-1))
'''
