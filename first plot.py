# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:55:01 2024

@author: David
"""

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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
ax.plot(x_new,gauss_function(x_new,popt1[0],popt1[1],popt1[2],popt1[3]),label="Fit")
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
ax.plot(x_new,gauss_function(x_new,popt2[0],popt2[1],popt2[2],popt2[3]),label="Fit")
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
ax.plot(x_new,gauss_function(x_new,popt3[0],popt3[1],popt3[2],popt3[3]),label="Fit")
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
ax.plot(x_new,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3]),label="Fit")
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

#----------------------------------------------------------------------------

fabry_perot = results[4]

y_centres = np.array([])
x_centres = np.array([])

count=0
for i, val in enumerate(fabry_perot):
    if val >= 0.05 and i >= count:
        for j,j_val in enumerate(fabry_perot[i:]):
            if j_val <= 0.05:
                # this came to me in a dream :)
                x_centre, y_centre = max(zip(x[i:i+j], np.convolve(fabry_perot,np.ones(10)/10,mode="same")[i:i+j]), key=lambda l: l[1])

                y_centres = np.append(y_centres,y_centre)
                x_centres = np.append(x_centres,x_centre)
                count = i + j
                break 
    else:
        continue

fabry_diffs = np.ediff1d(x_centres)
x_diffs = np.arange(0,len(fabry_diffs))
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_diffs,fabry_diffs,color="red",label="Difference in Fabry-Perot Peaks")
plt.show()

print("Differences:", x_centres)

fsr = (3e8) / 0.8
count=0
axis = np.array([])
for i, val in enumerate(x_centres[:-1]):
    segment = np.array([])
    for j in x:
        if j >= x_centres[i] and j < x_centres[i+1]:
            segment = np.append(segment,j)
    print("lower:",x_centres[i])
    print("Length of x segment:",len(segment))
    div = fsr/len(segment)
    new_segment = np.arange(0,fsr+div,div)[1:]
    new_segment = new_segment+(count*fsr)
    print("New segment:",new_segment)
    print("Length of new segment:",len(new_segment))
    print("upper:",x_centres[i+1])
    axis = np.append(axis, new_segment)
    count += 1
lower_index = [n for n,i in enumerate(x) if i>=x_centres[0]][0]
upper_index = [n for n,i in enumerate(x) if i<=x_centres[-1]][-1]
x_new = x[lower_index:upper_index+1]



# ------------------- THESE ARE THE FABRY-PEROT CORRECTED ARRAYS. DO NOT TOUCH!!!!!! ---------------------------------
freq = axis
dopper_free = C1[lower_index:upper_index+1]
dopper = C2[lower_index:upper_index+1]
fabry = fabry_perot[lower_index:upper_index+1]
difference = difference[lower_index:upper_index+1]

# ------------------- MAKING FANCY PLOT: ---------------------------------

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.set_xlabel('Time (s)', fontsize=18)
ax1.set_ylabel('Transmission', color="black",fontsize=18)
ax1.plot(x_new, dopper, color="black",label="Original Spectrum",linestyle="dotted")

ax1.tick_params(axis='x', labelcolor="black")
ax1.tick_params(axis='both', which='major', labelsize=15)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

ax2.set_xlabel('Frequency (Hz)', color="red", fontsize=18)  # we already handled the y-label with ax1
ax2.plot(axis, dopper, color="red",label="Corrected Spectrum",linestyle="dotted")
ax1.plot(x_new,fabry*33 - 7,color="black",label="Original FP")

def double_arrow(x1,y1,x2,y2,fsr=False):
    plt.arrow(x1,y1,x2-x1,y2-y1,color="green", head_length = 0.07e9, head_width = 0.14, length_includes_head = True)
    plt.arrow(x2,y2,x1-x2,y1-y2,color="green", head_length = 0.07e9, head_width = 0.14, length_includes_head = True)
    if fsr:
        plt.text((x1+x2)/2 - 0.3e9,y1+0.2,"FSR",color="green",fontsize=18)
    return None

# double_arrow(4.877e9,-5,5.253e9,-5)
double_arrow(6.375e9,-5,6.75e9,-5,fsr=True)
# double_arrow(5.625e9,-5,6e9,-5)
# double_arrow(6e9,-5,6.375e9,-5)
lns4=ax2.plot(axis, fabry*33 - 7,color="red",label="Corrected FP")

ax2.tick_params(axis='x', labelcolor="red")
ax2.tick_params(axis='both', which='major', labelsize=15)

fig.legend(loc=(0.7,0.45))

fig.tight_layout()  # otherwise the top x-label is slightly clipped
plt.show()

# --------------------------------------------------------------------------------------------------------------



