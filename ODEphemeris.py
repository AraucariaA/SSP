# OD EPHEMERIS:

import odlibrary.odlib as od
import numpy as np

r = [3.970631912189709E-01, -1.225073703123122E+00, 4.747425159692229E-01]
r_dot = [1.139883471649287E-02, 2.679831533677191E-03, 3.750852804158524E-03]

r, r_dot = od.gaussifyer(r,r_dot)



#2
def Ephemeris(r, r_dot):
    # a = od.SMAFinder(r, r_dot)
    # e = od.eccenFinder(r, r_dot)
    # w = od.littleOmegaFinder(r,r_dot)
    # i = od.inclinFinder(r, r_dot)
    # T = od.periPassFinder(r, r_dot)
    # bigO = od.bigOmegaFinder(r, r_dot)
    # Einit = od.newtonAnomFinder(r, r_dot)

    a = 1.056794818503845E+00
    e = 3.442356649108365E-01
    w = 4.459378
    i = 0.4390407
    bigO = 4.123126
    M = 2.767504





    #TODO: Change for the right date... does this mean I need another input? :(
    epsilon = 0.4090926 #radians
    k = 0.01720209894 
    R = [-6.57371E-01, 7.09258E-01, 3.0743E-01]



    #I might want to change that error?
    E = od.newtonsMeth(lambda Einit: M-(Einit-e*np.sin(Einit)), lambda Einit: -1+e*np.cos(Einit), M, 1e-12)
    
    # print("M", M)
    # print("E", E)

    xCart = a*np.cos(E) - a*e
    yCart = a*np.sqrt(1-(e**2))*np.sin(E)
    zCart = 0

    coordsARR = np.array([[[xCart],[yCart],[zCart]]])
    coordsLST = [xCart, yCart, zCart]
    # print("Cartesian1-1", coordsLST)


    ###########################################################################################################################################################


    matrixR = np.array([[np.cos(w), -1*np.sin(w), 0],
                        [np.sin(w), np.cos(w), 0],
                        [0,0,1]])
    matrixM = np.array([[1, 0,0],
                        [0, np.cos(i), -1*np.sin(i)],
                        [0, np.sin(i), np.cos(i)]])
    matrixL = np.array([[np.cos(bigO), -1*np.sin(bigO), 0],
                        [np.sin(bigO), np.cos(bigO), 0],
                        [0,0,1]])
    
    prod1 = np.matmul(matrixL, matrixM)
    prod2 = np.matmul(prod1, matrixR)
    eclipticCoords = np.matmul(prod2, coordsARR)
    
    
    matrix1 = np.array([[1,0,0],
                        [0, np.cos(epsilon), -1*np.sin(epsilon)],
                        [0, np.sin(epsilon), np.cos(epsilon)]])

    EquatCoords = np.matmul(matrix1, eclipticCoords)


    ###########################################################################################################################################################


    rho = []
    rhoHat = []
    for j in range(0,3):
        item = R[j] + EquatCoords[0][j][0]
        rho.append(item)
    rhoMag = od.magFinder(rho)

    for i in range(0,3):
        item = rho[i]/rhoMag
        rhoHat.append(item)

    DEC = np.arcsin(rhoHat[2])
    cosRA = (rhoHat[0])/(np.cos(DEC))
    sinRA = (rhoHat[1])/(np.cos(DEC))

    RA = od.quadamb(sinRA, cosRA)

    RAdeg = np.rad2deg(RA)
    DECdeg = np.rad2deg(DEC)
    # print("Equat", EquatCoords)
    return RAdeg, DECdeg





# Test Cases:
print("RA and DEC", Ephemeris(r,r_dot))