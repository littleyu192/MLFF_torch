# import matplotlib.pyplot as plt
import numpy as np
import os


filename1="E_fit"
data1 = np.loadtxt(filename1)#, delimiter='  ')
error1=np.sum((data1[:,2]-data1[:,1])**2,axis=0)#/len(data1)

filename2="energyL.pred.1"
data2 = np.loadtxt(filename2)#, delimiter='  ')
error2=np.sum((data2[:,0]-data2[:,1])**2,axis=0)#/len(data2)

print('MSE of GPR fit:',error1,'\n'+'MSE of linear fit:',error2)
