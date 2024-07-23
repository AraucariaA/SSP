import numpy as np
import odlibrary.odlib as od


def rhoFinder(RA, DEC, switch):
    '''
    switch = 0 means that you want to input the RA and DEC as lists
    switch = 1 means that you want to input the RA and DEC already decimalized in radians
    '''
    if switch == 0:
        RArad = od.HMSconverter(RA[0], RA[1], RA[2], 1)
        DECdeg = od.DMStoDeg(DEC[0], DEC[1], DEC[2])
        DECrad = np.deg2rad(DECdeg)
    else:
        RA = RArad
        DEC = DECrad

    rhoHat_z = np.sin(DECrad)
    rhoHat_x = np.cos(RArad) * np.cos(DECrad)
    rhoHat_y = np.cos(DECrad)*np.sin(RArad)

    rhoHat = [rhoHat_x, rhoHat_y, rhoHat_z]
    return rhoHat


def DxxFinder(rho1, R1, rho2, R2, rho3, R3):
    D0 = od.dot(od.cross(rho2, rho3), rho1)

    D11 = od.dot(od.cross(R1, rho2), rho3)
    D12 = od.dot(od.cross(R2, rho2), rho3)
    D13 = od.dot(od.cross(R3, rho2), rho3)

    D21 = od.dot(od.cross(rho1, R1), rho3)
    D22 = od.dot(od.cross(rho1, R2), rho3)
    D23 = od.dot(od.cross(rho1, R3), rho3)

    D31 = od.dot(rho1, od.cross(rho2, R1))
    D32 = od.dot(rho1, od.cross(rho2, R2))
    D33 = od.dot(rho1, od.cross(rho2, R3))

    # D0 = od.dot(rho1, od.cross(rho3, rho2))
    # D11 = od.dot(od.cross(rho2, R1), rho3)
    # D12 = od.dot(od.cross(rho2, R2), rho3)
    # D13 = od.dot(od.cross(rho2, R3), rho3)

    # D21 = od.dot(od.cross(R1, rho1), rho3)
    # D22 = od.dot(od.cross(R2, rho1), rho3)
    # D23 = od.dot(od.cross(R3, rho1), rho3)

    # D31 = od.dot(rho1, od.cross(R1, rho2))
    # D32 = od.dot(rho1, od.cross(R2, rho2))
    # D33 = od.dot(rho1, od.cross(R3, rho2))



    allvals = [D0, D11, D12, D13, D21, D22, D23, D31, D32, D33]
    return allvals


set1 = ["2021 06 25", "00:00:00.000", 18,25,08.44, -17,26,41.3, -5.985728598861461E-02,
9.309676159065817E-01, 4.035414693476737E-01]

set2 = ["2021 07 05", "00:00:00.000", 18,15,28.85, -16,27,16.5, -2.271502585826002E-01,
9.092709064712199E-01, 3.941342306093848E-01]

set3 = ["2021 07 15", "00:00:00.000", 18,5,40.89, -15,30,48.9, -3.881336047533506E-01,
8.619617590425438E-01, 3.736284118981542E-01]

def builder(set1, set2, set3):
    #Parsing
    RA1 = [set1[2], set1[3], set1[4]]
    DEC1 = [set1[5], set1[6], set1[7]]
    R1 = [set1[8], set1[9], set1[10]]

    RA2 = [set2[2], set2[3], set2[4]]
    DEC2 = [set2[5], set2[6], set2[7]]
    R2 = [set2[8], set2[9], set2[10]]

    RA3 = [set3[2], set3[3], set3[4]]
    DEC3 = [set3[5], set3[6], set3[7]]
    R3 = [set3[8], set3[9], set3[10]]

    #Finding the rho's
    rho1 = rhoFinder(RA1, DEC1, 0)
    rho2 = rhoFinder(RA2, DEC2, 0)
    rho3 = rhoFinder(RA3, DEC3, 0)

    print("rho1", rho1)
    print("rho2", rho2)
    print("rho3", rho3)


    allvals = DxxFinder(rho1, R1, rho2, R2, rho3, R3)

    for i in range(0, len(allvals)):
        print("D", i%3, "  :  ", allvals[i])


print(builder(set1, set2, set3))
