import numpy as np
from matplotlib import pyplot as plt
import random


#PS6-2
#A
def planeExplorer(N, r, switch):
    '''
    switch = 0 turns off graphing
    switch = 1 turns on graphing
    '''
    xCoords = [0]
    yCoords = [0]

    for i in range(1, N+1):
        #Choose a direction
        angle = random.uniform(0, 360)

        xstep = xCoords[i-1] + r*np.cos(angle)
        ystep = yCoords[i-1] + r*np.sin(angle)
        xCoords.append(xstep)
        yCoords.append(ystep)

    if switch == 1:
        fig, ax = plt.subplots()
        ax.scatter(xCoords, yCoords)
        ax.plot(xCoords, yCoords)
        plt.show()
    
    dist = np.sqrt(xCoords[-1]**2 +yCoords[-1]**2)
    return dist

#B
#and C
def distAvg(samples, N, r):
    sumA = 0
    sumB = 0
    positions = []
    for i in range(0, samples):
        item = planeExplorer(N,r,0)
        positions.append(item)
        sumA += item
        sumB += item**2
    rmsC = np.sqrt(sumB/len(positions))
    plt.hist(positions, bins=30)
    plt.show()
    average = sumA/samples
    rmsP = np.sqrt(N) * r
    return average, rmsC

print(distAvg(10000, 100, 1))




def centroidFinder(filepath):

    # "/home/proctort/Documents/PSETs/InputsandResults/centroidData.csv"
    with open(str(filepath)) as file:
        entireArray = file.readlines()

        data = []

        for i in range(0, len(entireArray)):
            row = entireArray[i].split(",")
            row = entireArray[i].split("/n")
            row = entireArray[i].split(",")
            for j in range(0, len(row)):
                row[j] = float(row[j])

            data.append(row)
    
    width = len(data[0])
    height = len(data)

    xCM_top = 0
    yCM_top = 0

    N = 0
    for i in range(0, height):
        for j in range(0, width):
            xCM_top += data[i][j]*j
            yCM_top += data[i][j]*i
            N += data[i][j]
    xCM = xCM_top/N
    yCM = yCM_top/N

    xCM_delta = 0
    predeltX = 0
    predeltY = 0
    for i in range(0, height):
        for j in range(0, width):
            predeltX += ((j-xCM)**2 )*(data[i][j])
            predeltY += ((i-yCM)**2 )*(data[i][j])

    xCM_delta = (1/N)*np.sqrt(predeltX)
    yCM_delta = (1/N)*np.sqrt(predeltY)

    # Displaying the data:

    map = plt.imshow(data) #M is your np.array of the photon counts
    plt.colorbar(map)
    plt.plot(xCM, yCM,'+') #Xcm, Ycm are centroid coordinates
    plt.show()

    return (xCM, yCM)
        

# Testing:
# print(centroidFinder("/home/proctort/Documents/PSETs/InputsandResults/centroidData.csv"))