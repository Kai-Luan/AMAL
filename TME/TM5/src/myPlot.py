import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bbrl_algos.rliable_stats.tests import run_test
import os

def aggreg(path):
    X = []
    for data_file in os.listdir(path):
        filepath = path + data_file+'/dqn_data/dqn_LunarLander-v2.data' 
        X.append(np.loadtxt(filepath).mean(0))
    X = np.vstack(X)
    return X

font = {}
matplotlib.rc("font", **font)
sys.path.append("../")


save = False  # save in ./plot.png if True


dqn_perfs = aggreg("/users/Etu5/28601285/Documents/data/DQN/")
ddqn_perfs = aggreg("/users/Etu5/28601285/Documents/data/DDQN/")

ax = plt.plot(ddqn_perfs.T.mean(1), label= 'DDQN', color='#ff7f0e',alpha=0.5)
plt.setp(ax[1:], label="_")
ax = plt.plot(dqn_perfs.T.mean(1), label= 'DQN', color='#1f77b4', alpha=0.5)
plt.setp(ax[1:], label="_")

lab1 = plt.xlabel("training steps")
lab2 = plt.ylabel("performance")
plt.legend()