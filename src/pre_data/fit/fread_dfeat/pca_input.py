#!/usr/bin/env python3
#-*- coding utf-8 -*-

import sys
import random
fp='fit.input'
with open(fp,'r') as f:
    line=f.readline()
    nType=int(line.split(',')[0])
    line=f.readline()
    line=f.readline()
    nFeat2=int(line.split(',')[1].split()[0])

iType=int(sys.argv[1])

print(iType)
#print(nFeat2)
print(random.randint(-2100000000,-1))
