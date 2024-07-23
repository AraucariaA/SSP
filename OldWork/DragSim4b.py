import PSET3
import numpy as np
from matplotlib import pyplot as plt


# rMag = int(input("rMag"))
# theta = int(input("theta"))
# xZero = int(input("xZero"))
# yZero = int(input("yZero"))

rMag = 30
theta = 60
xZero = 0
yZero = 0

g = -9.81
C=1/2
aDensity = 1.4
radius = 1/6 #in meters
crossArea = np.pi * pow(radius,2)
ftbMass = 0.415 #in kg
velo1 = 0
b = 0.002
deltaT = 0.01

# Up is positive
# Down is negative

radtheta = np.deg2rad(theta)

v_x = np.cos(theta) * rMag
v_y = np.sin(theta) * rMag

xPoints = ([xZero])
yPoints = ([yZero])

counter = 0
while yPoints[counter]>0:
    velocity = np.sqrt(pow(v_x,2)+pow(v_y,2))
    alpha = np.arctan(v_y/v_x) # alpha is the angle from the horizontal to the fDrag vector

    acelDrag = b*pow(velocity,2)/ftbMass
    xADrag = acelDrag * np.cos(alpha)
    yADrag = acelDrag * np.sin(alpha)

    xAcel = xADrag
    yAcel = g + yADrag

    deltX = v_x*deltaT + (1/2)*xAcel*(pow(deltaT,2))
    deltY = v_y*deltaT + (1/2)*yAcel*(pow(deltaT,2))

    xNew = xPoints[counter] + deltX
    xPoints = np.append(xPoints, xNew)

    yNew = yPoints[counter] + deltY
    yPoints = np.append(yPoints, yNew)


    counter+=1




fig, ax = plt.subplots()

ax.plot(xPoints, yPoints)

plt.show()