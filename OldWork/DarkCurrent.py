import numpy as np
from matplotlib import pyplot as plt
import PSET4_7

#Dark vs CMOS

stdCMOS = [18.431, 15.932, 14.479, 12.89, 11.17]
medCMOS = [63, 62, 62, 62, 62]
tempCMOS = [9.9, 4.3, 0, -4.7, -9.1]

m1, b1 = PSET4_7.linReg(tempCMOS, medCMOS)

print("Slope of Temp Graph:", m1)
print("Y-Int of Temp Graph:", b1)

bf1 = [min(tempCMOS), max(tempCMOS)]
bestfit1lin = [m1*bf1[0]+b1, m1*bf1[1]+b1]

model = np.poly1d(np.polyfit(tempCMOS, medCMOS, 2))
print("Quad equation:", model)



#GRAPHING STUFF
fig1, ax1 = plt.subplots()
plt.scatter(tempCMOS,medCMOS)

plt.errorbar(tempCMOS, medCMOS, stdCMOS, xerr=None, fmt='',
             ecolor=None, elinewidth=None, capsize=20, 
             barsabove=True, lolims=False, uplims=False, 
             xlolims=False, xuplims=False, errorevery=1, 
             capthick=None, data=None)

ax1.set_title("Dark Current as a Function of CMOS Temperature")
ax1.set_xlabel("CMOS Internal Temperature (Celcius)")
ax1.set_ylabel("Median Image Counts")

ax1.plot(bf1, bestfit1lin, color = "rebeccapurple")
ax1.plot(bf1, model(bf1), color = "green")






######################################################################################################
######################################################################################################
######################################################################################################






#Dark vs Exposure Time
stdEXP = [3.178, 14.71, 17.53, 19.583, 20.677, 21.733, 23.252, 24.319, 25.617]
medEXP = [61, 62, 63, 64, 64, 65, 66, 66, 67]
timeEXP = [1, 25, 50, 75, 90, 108, 134, 155, 180]

m2, b2 = PSET4_7.linReg(timeEXP, medEXP)
print("Slope of EXP Graph:", m2)
print("Y-Int of EXP Graph:", b2)

bf2l = [min(timeEXP), max(timeEXP)]
bestfit2lin = [m2*bf2l[0]+b2, m2*bf2l[1]+b2]



#GRAPHING STUFF
fig2, ax2 = plt.subplots()
ax2.scatter(timeEXP,medEXP)

plt.errorbar(timeEXP, medEXP, stdEXP, xerr=None, fmt='',
             ecolor=None, elinewidth=None, capsize=20, 
             barsabove=True, lolims=False, uplims=False, 
             xlolims=False, xuplims=False, errorevery=1, 
             capthick=None, data=None)

ax2.set_title("Dark Current as a Function of Exposure Time")
ax2.set_xlabel("Exposure Time (Seconds)")
ax2.set_ylabel("Median Image Counts")


ax2.plot(bf2l, bestfit2lin, color = "red")




plt.show()