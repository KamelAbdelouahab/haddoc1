import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color': 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'font.size': 15,
    'axes.facecolor': 'white'
}
rcParams.update(params)

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

data = np.genfromtxt('./exploration.csv', delimiter=',')
n1_index  = 0;
n2_index  = 1;
n3_index  = 2;
bw_index  = 3;
tpr_index = 4;
dsp_index = 6;
tdr_index = 7;

dsp = data[:, dsp_index]
tpr = data[:, tpr_index]

colors = ['g', 'b', 'r', 'c', 'm']
i = 0
plt.figure(figsize=(6, 5))
plt.grid()
for bw in range(3,8):
    tpr_bw = data[np.where(data[:,bw_index]==bw), tpr_index]
    dsp_bw = data[np.where(data[:,bw_index]==bw), dsp_index]
    plt.scatter(dsp_bw, tpr_bw, marker='o', edgecolor='k', color=colors[i], alpha=0.7)
    i +=1


p_front = pareto_frontier(dsp, tpr, maxX = False, maxY = True)
plt.plot(p_front[0], p_front[1], '--b')
plt.xlabel("DSP Blocks Instanciated")
plt.ylabel("TPR (%)")
plt.legend(['Paratto','3 bits', '4 bits', '5 bits', '6 bits', '7 bits'])
#plt.show()
plt.savefig("./holo-pareto.pdf", bbox_inches ='tight')
