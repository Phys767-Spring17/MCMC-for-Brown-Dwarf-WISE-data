import mcmc_clone_emceee
import time
import runpy
import os
import subprocess
import timeit
import numpy as np
from timeit import Timer


def fun():
    proc = subprocess.Popen(["python","mcmc_clone_emceee0"], shell = True)
    proc.terminate()

def fun2():
    proc = subprocess.Popen(["python","mcmc_clone_emceee"], shell = True)
    proc.terminate()

t = Timer(lambda: fun())

t2 = Timer(lambda: fun2())
t_list = []
t2_list = []
for i in range(1000):
    t2_list.append(int(t2.timeit(number=1)*10**10))
    t_list.append(int(t.timeit(number=1)*10**10))

t_list_B = []
t2_list_B = []
for a in t_list:
    t_list_B.append(a/10**10)

for b in t2_list:
    t2_list_B.append(b/10**10)

# for x in t_list_B:
#    print(x)
print("----------------------------------------------------")
print(np.average((t_list_B)),"MCMC process bar included")
print(" ")

#for x in t2_list_B:
#    print(x)
print("----------------------------------------------------")
print(np.average((t2_list_B)),"MCMC standard")
print(" ")

print("----------------------------------------------------")
print("Implementating classes on average,increases the runtime by",np.average((t2_list_B))-np.average((t_list_B)),"ms")
