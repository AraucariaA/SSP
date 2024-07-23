import PSET3
import numpy as np
from matplotlib import pyplot as plt



def ProjSim4a(rMag, theta, xZero, yZero):
    '''
    rMag: initial velocity
    theta: angle from the horizontal of velocity vector
    xZero: x-coordinate from which you wish to start
    yZero: y-coordinate from which you wish to start
    '''
    g = -9.81
    fig, ax = plt.subplots()

    end = PSET3.rangeFinder(rMag, theta, xZero, yZero)
    peak = PSET3.peakFinder(rMag, theta, xZero, yZero)

    def yPos(t):
        yPos = yZero + (np.sin(np.deg2rad(theta)))*rMag*(t) + (g/2) * (pow(t,2))
        return yPos

    xVals = []
    yVals = []
    for i in range(0, int(np.ceil(end))):
        i = i/5
        if yPos(i) < 0:
            break
        
        xVals.append(i)
        yVals.append(yPos(i))

    ax.plot(xVals, yVals)

    plt.show()

