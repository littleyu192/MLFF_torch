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
    line=f.readline()
    nRef=int(line.split()[0])

iType=int(sys.argv[1])

print(iType)
print(nRef)
print(random.randint(-2100000000,-1))
