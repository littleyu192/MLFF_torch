import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

var = np.loadtxt('Gfeat_fit_var')

f=open('plot','w')

for i in range(len(var)):
    if float(var[i,0])<0.001:
        f.writelines(str(var[i,0])+'  '+str(var[i,1])+'\n')
    else:
        continue
f.close

plot = np.loadtxt('plot')

plt.figure()
plt.scatter(float(plot[:,0]), float(plot[:,1]))
    # dotetest = plt.scatter(etest[:,0], etest[:,1])
    # exey = plt.plot([etrain[:,0].min(),etrain[:,0].max()], [etrain[:,0].min(),etrain[:,0].max()], ls='-', color='C2')
    # plt.legend([dotetrain, dotetest],
    #            ['train '+str(ele), 'test '+str(ele)])

    # plt.figure()
    # dotftrain = plt.scatter(ftrain_real, ftrain_pred)
    # dotftest = plt.scatter(ftest_real, ftest_pred)
    # exey = plt.plot([ftrain_real.min(),ftrain_real.max()], [ftrain_real.min(),ftrain_real.max()], ls='-', color='C2')
    # plt.legend([dotftrain, dotftest],
    #            ['train '+str(ele), 'test '+str(ele)])


plt.show()
