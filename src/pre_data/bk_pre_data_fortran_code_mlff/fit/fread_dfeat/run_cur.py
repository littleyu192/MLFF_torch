#!/usr/bin/env python3
#-*- coding utf-8 -*-

import os
import sys
import random
fp='fit.input'
with open(fp,'r') as f:
    line=f.readline()
    nType=int(line.split(',')[0])
    nRef=[]
    for i in range(nType):
        line=f.readline()
        nRef.append(int(line.split(',')[3].split()[0]))
        strLines=str(i+1)+' \n'+str(nRef[i])+' \n'+str(random.randint(-2100000000,-1))+' \n'
        command='echo "'+strLines+'" | ../sparcify_CUR2.r'
        #print(strLines)
        #print(command)
        os.system(command)
'''
iType=int(sys.argv[1])

print(iType)
print(nRef[iType-1])
print(random.randint(-2100000000,-1))
'''
