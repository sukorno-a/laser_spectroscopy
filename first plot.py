# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:55:01 2024

@author: David
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl

def gauss_function(x, a, x0, sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

file = "P200P10.csv"
results = pd.read_csv(file, sep = ',', header=None)

results = results.loc[(results[0] > 0.068) & (results[0] < 0.11)]
x = results[0]
C1 = results[1]
C2 = results[2]
difference = results[3]

fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x,C1,color="Orange",label="Channel 1")
# ax.plot(x,C2,color="Blue",label="Channel 2")
ax.plot(x,difference,color="Orange",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

##FIRST PEAK CORRESPONDS TO 87Rb F=2
first_peak = results.loc[(results[0] > 0.075) & (results[0] < 0.079)]

popt1, pcov1 = sp.optimize.curve_fit(gauss_function, first_peak[0], first_peak[1], p0 = [-1, -0.0155, 0.01,-0.05],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
x_new=np.linspace(0.075,0.079,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt1[0],popt1[1],popt1[2],popt1[3]))
#ax.plot(first_peak[0],first_peak[1],color="Orange",label="Channel 1")
#ax.plot(first_peak[0],first_peak[2],color="Blue",label="Channel 2")
ax.plot(first_peak[0],first_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


##FIRST PEAK CORRESPONDS TO 85Rb F=3
second_peak = results.loc[(results[0] > 0.0815) & (results[0] < 0.0835)]

popt2, pcov2 = sp.optimize.curve_fit(gauss_function, second_peak[0], second_peak[1], p0 = [-1, -0.01, 0.01,0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
x_new=np.linspace(0.0815,0.0835,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt2[0],popt2[1],popt2[2],popt2[3]))
# ax.plot(second_peak[0],second_peak[1],color="Orange",label="Channel 1")
# ax.plot(second_peak[0],second_peak[2],color="Blue",label="Channel 2")
ax.plot(second_peak[0],second_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


##FIRST PEAK CORRESPONDS TO 85Rb F=2
third_peak = results.loc[(results[0] > 0.0945) & (results[0] < 0.0955)]

popt3, pcov3 = sp.optimize.curve_fit(gauss_function, third_peak[0], third_peak[1], p0 = [-1, 0.0015, 0.01, 0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
x_new=np.linspace(0.0945,0.0955,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt3[0],popt3[1],popt3[2],popt3[3]))
# ax.plot(third_peak[0],third_peak[1],color="Orange",label="Channel 1")
# ax.plot(third_peak[0],third_peak[2],color="Blue",label="Channel 2")
ax.plot(third_peak[0],third_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


##FIRST PEAK CORRESPONDS TO 87Rb F=1
fourth_peak = results.loc[(results[0] > 0.104) & (results[0] < 0.106)]

popt4, pcov4 = sp.optimize.curve_fit(gauss_function, fourth_peak[0], fourth_peak[1], p0 = [-1, 0.0175, 0.001, 0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
x_new=np.linspace(0.104,0.106,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3]))
# ax.plot(fourth_peak[0],fourth_peak[1],color="Orange",label="Channel 1")
# ax.plot(fourth_peak[0],fourth_peak[2],color="Blue",label="Channel 2")
ax.plot(fourth_peak[0],fourth_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()