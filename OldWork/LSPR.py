#LSPR

from matplotlib import pyplot as plt
import numpy as np
import PSET1
import PSET2


def cramers(matBig, matC, j):
    '''
    mat1 is a 3x3 matrix
    mat2 is the constanct column vector
    j is the column being replaced, 0 being the first column
    '''
    matrix = np.copy(matBig)

    D = np.linalg.det(matBig)


    for i in range(0,len(matBig)):
        matrix[i][j] = matC[i][0]
    

    Dreplaced = np.linalg.det(matrix)

    # print("D: ", D)
    # print("Dreplaced: ", Dreplaced)
    return (Dreplaced/D)

def LSPR(filepath):

    # "/home/proctort/Documents/PSETs/LSPR_test_input.csv"
    with open(str(filepath)) as file:
        entireArray = file.readlines()

        preData = []

        for i in range(1, len(entireArray)):
            row = entireArray[i].split(",")
            row = entireArray[i].split("/n")
            row = entireArray[i].split(",")
            for j in range(0, len(row)):
                try:
                    row[j] = float(row[j])
                except:
                    row[j] = float(0)
            preData.append(row)
        
        # data[0] is full of strings corresponding to positions
        dataC = []

        for i in range(0,8):
            newCol = []
            for j in range(0, len(preData)):
                newCol.append(preData[j][i])
            dataC.append(newCol)
   
    N = len(dataC[0]) - 1 #This is minus one because I still have the asteroid in the data set and I only want to reference stars.

    #Now I have this all in columns, onto actually using it.

    for j in range(0, N):
        RAdeg = PSET1.HMSconverter(dataC[2][j], dataC[3][j], dataC[4][j], 0)
        DECdeg = PSET2.DMStoDeg(dataC[5][j], dataC[6][j], dataC[7][j])

    # Prepping the matrix for Cramer's:

    sumX = 0
    sumY = 0
    sumX2 = 0
    sumY2 = 0
    sumXY = 0

    sumRA = 0
    sumRAX = 0
    sumRAY = 0

    sumDEC = 0
    sumDECX = 0
    sumDECY = 0

    for i in range(N):
        sumX += dataC[0][i]
        sumY += dataC[1][i]
        sumX2 += pow(dataC[0][i],2)
        sumY2 += pow(dataC[1][i],2)
        sumXY += dataC[0][i] * dataC[1][i]

        RA = PSET1.HMSconverter(dataC[2][i], dataC[3][i], dataC[4][i], 0)
        sumRA += RA
        sumRAX += RA * dataC[0][i]
        sumRAY += RA * dataC[1][i]

        DEC = PSET2.DMStoDeg(dataC[5][i], dataC[6][i], dataC[7][i])
        sumDEC += DEC
        sumDECX += DEC * dataC[0][i]
        sumDECY += DEC * dataC[1][i]

    bigArr = np.array([[N, sumX, sumY],
                        [sumX, sumX2, sumXY],
                        [sumY, sumXY, sumY2]])

    colArrRA = np.array([[sumRA], [sumRAX], [sumRAY]])
    colArrDEC = np.array([[sumDEC], [sumDECX], [sumDECY]])

    b1 = cramers(bigArr, colArrRA, 0)
    a11 = cramers(bigArr, colArrRA, 1)
    a12 = cramers(bigArr, colArrRA, 2)

    b2 = cramers(bigArr, colArrDEC, 0)
    a21 = cramers(bigArr, colArrDEC, 1)
    a22 = cramers(bigArr, colArrDEC, 2)

    # print("b1: ", b1)
    # print("a11: ", a11)
    # print("a12: ", a12)
    # print("b2: ", b2)
    # print("a21: ", a21)
    # print("a22: ", a22)

    preSigmaRA = 0
    preSigmaDEC = 0

    for i in range(0, N):
        RA1 = PSET1.HMSconverter(dataC[2][i], dataC[3][i], dataC[4][i], 0)
        DEC1 = PSET2.DMStoDeg(dataC[5][i], dataC[6][i], dataC[7][i])

        preSigmaRA += (RA1 - (b1 + a11*dataC[0][i] + a12*dataC[1][i]))**2
        preSigmaDEC += (DEC1 - (b2 + a21*dataC[0][i] + a22*dataC[1][i]))**2

    sigmaRA = (np.sqrt((1/(N-3)) * preSigmaRA))*3600
    sigmaDEC = (np.sqrt((1/(N-3)) * preSigmaDEC))*3600


    asteroidRA = b1 + a11*dataC[0][-1] + a12*dataC[1][-1]
    asteroidRA = asteroidRA/15
    asteroidDEC = b2 + a21*dataC[0][-1] + a22*dataC[1][-1]

    # print("asteroidRA", asteroidRA)
    # print("asteroidDEC", asteroidDEC)

    magList = [9.12, 8.51, 9.96, 8.47, 9.29, 10.77]
    # fluxList = []

    # for i in range(0, len(magList)):
    #     f = 10**((48-magList[i])/2.5)
    #     fluxList.append(f)


    for i in range(0, N):
        plt.scatter(dataC[0][i], dataC[1][i], c="orange", linewidths= (5000/(magList[i]**3)))

    plt.scatter(dataC[0][-1], dataC[1][-1], c="black")

    plt.show()

    return(b1, b2, a11, a12, a21, a22, asteroidRA, asteroidDEC, sigmaRA, sigmaDEC)




print(LSPR("/home/proctort/Documents/PSETs/InputsandResults/LSPR_test_input.csv"))