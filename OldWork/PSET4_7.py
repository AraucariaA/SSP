#PSET - 4
#PS4-7

from matplotlib import pyplot as plt
import numpy as np



def linReg(xCoords, yCoords):
    '''
    Input all your x-coordinates as a list for the first input,
    and then give all your y-coordinates as a list for the second input
    This returns the slope and y-intercept of the linear regression line
    Uncomment the code to have it show something.
    '''
    sumX = 0
    sumY = 0
    sumXY = 0
    sumX2 = 0

    N = len(xCoords)

    for i in range(N):
        sumXY += (xCoords[i]) * (yCoords[i])
        sumX += xCoords[i]
        sumY += yCoords[i]
        sumX2 += pow(xCoords[i], 2)


    mTop = sumXY*N - sumX*sumY
    mBottom = sumX2*N - sumX*sumX

    m = mTop/mBottom

    bTop = sumX2*sumY - sumXY*sumX
    bBottom = sumX2*N - sumX*sumX

    b = bTop/bBottom

    return (m, b)

    #Now that I have my line I need to populate an array with the values

    # xVals = [min(xCoords)-2, max(xCoords)+2]
    # yVals = [m*xVals[0]+b, m*xVals[1]+b]



    #GRAPH SETUP

    # plt.xlim(min(xCoords)-1, max(xCoords)+1)
    # plt.ylim(min(yCoords)-1, max(yCoords)+1)
    # plt.scatter(xCoords, yCoords)


    # plt.plot(xVals, yVals)

    # plt.show()

