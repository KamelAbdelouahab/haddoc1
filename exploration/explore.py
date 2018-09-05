import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Garamond"
from matplotlib import rcParams
params = {
   'font.family': 'Garamond',
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'figure.figsize': [5.5, 4],
   'axes.facecolor' : 'white'
    #'text.usetex'    : 'true'
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
nb_index  = 3;
tpr_index = 4;
dsp_index = 6;
tdr_index = 7;

tpr = data[:,tpr_index]
dsp = data[:,dsp_index]

p_front = pareto_frontier(dsp, tpr, maxX = False, maxY = True)

plt.scatter(dsp, tpr, marker='o')
plt.plot(p_front[0], p_front[1])
plt.grid(linestyle="dotted")
plt.xlabel("DSP Blocks Instanciated")
plt.ylabel("TPR (%)")
plt.show()
#plt.savefig("C:/Users/Kamel/Documents/PhD/Manuscript/Figures/Haddoc1-Res/holo-pareto.pdf", bbox_inches ='tight')
