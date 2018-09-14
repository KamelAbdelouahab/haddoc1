#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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
    'axes.labelsize': 15,
    'font.size': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.facecolor': 'white'
}
rcParams.update(params)

if __name__ == '__main__':
    # Fitting Results for various Implementations of LeNet5
    # ALM = [conv pool fifo etc ]

    I1 = np.array([245618, 2850, 119078, 86417])
    I2 = np.array([98830, 3020, 79355, 79446])
    I3 = np.array([207654, 5216, 164860, 188301])
    I4 = np.array([105924, 3574, 87885, 87995])

    plt.axis('equal')
    plt.figure(figsize=(15, 3.5))
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.arange(4)*4)

    explode = (0, 0, 0.1, 0)
    ax1 = plt.subplot(1,4,1)
    ax2 = plt.subplot(1,4,2)
    ax3 = plt.subplot(1,4,3)
    ax4 = plt.subplot(1,4,4)

    ax1.pie(100 * I1 / np.sum(I1), explode = explode, autopct='%1.1f%%', colors=colors)
    ax2.pie(100 * I2 / np.sum(I2), explode = explode, autopct='%1.1f%%', colors=colors)
    ax3.pie(100 * I3 / np.sum(I3), explode = explode, autopct='%1.1f%%', colors=colors)
    ax4.pie(100 * I4 / np.sum(I4), explode = explode, autopct='%1.1f%%', colors=colors)

    ax1.set_title('I1')
    ax2.set_title('I2')
    ax3.set_title('I3')
    ax4.set_title('I4')
    plt.legend(['conv', 'pool', 'FIFO', 'etc'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("ALMPie.pdf", bbox_inches ='tight')
