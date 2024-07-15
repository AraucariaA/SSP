#Gauss' Meth
import numpy as np
import odlib as od



# The input file should be formatted such that it contains one line for every usable night of data with the following info:
# • UT date/time at middle of observation (JD),
# • RA (hh:mm:ss.ss), Dec (dd:mm:ss.s), and
# • Sun vector from JPL Horizons (AU, equatorial cartesian, J2000, apparent
# states)

# E.g. file = []

#"/home/proctort/Documents/PSETs/InputsandResults/ProctorInputTest.txt"
def gaussMeth(filepath):
    '''
    This function outputs:
        - The asteroid's position and velocity vectors for the central observation in
            ecliptic rectangular coordinates, in AU and AU/day
        - The range (i.e. rho) to the asteroid for the central observation, in AU
        - The 6 orbital elements, all angles in degrees with respect to the ecliptic;
            report Mean anomaly for the date July 20, 2024 6:00 UT.
    '''

    #Parsing (this doesn't all need to be in the with statement, but I find that the organization is nicer this way):
    with open(str(filepath)) as file:
        entireArray = file.readlines()

        preData = []

        for i in range(1, len(entireArray)):
            row = entireArray[i].split("/n")
            row = entireArray[i].split(",")
            # row = entireArray[i].split(" ")
            # row = entireArray[i].split(":")
            
            for j in range(0, len(row)):
                try:
                    row[j] = float(row[j])
                except:
                    row[j] = float(0)
            preData.append(row)
        
        data = [[],[],[]]

        for i in range(0, len(preData)):
            #This is to turn the data from Gregorian to JD
            gregD = preData[i][0].split("")
            UTC = preData[i][1].split(":")
            JD = od.gregtoJD(gregD[0], gregD[1], gregD[2], UTC[0], UTC[1], UTC[2])
            GD = JD*58.13244086 #Converts to Gaussian days
            data[i].append(GD)

            #Decimalizing RA:
            preRA = preData[i][2].split(":")
            RAhrs = preRA[0]
            RAmin = preRA[1]
            RAsec = preRA[2]
            RArad = od.HMSconverter(RAhrs, RAmin, RAsec, 1)
            data[i].append(RArad)


            #Decimalizing DEC:
            preDEC = preData[i][3].split(":")
            DECdeg = preDEC[0]
            DECmin = preDEC[1]
            DECsec = preDEC[2]
            DECdecim = od.DMStoDeg(DECdeg, DECmin, DECsec)
            DECrad = np.deg2rad(DECdecim)
            data[i].append(DECrad)

            #Making the Sun Vector (R) its own list:
            x = preData[i][4]
            y = preData[i][5]
            z = preData[i][6]
            R = [x, y, z]
            data[i].append(R)
    
    # data is something like:
    # data = [[(Gaussian time), (RA in radians), (DEC in radians), [Sun Vector]],
                # [again for the second row],
                # [and for the third]]

    # LET'S FIND RHO 1, 2, AND 3

    #Since we're operating in AU and Gaussian days, mu = k^2 = GM = 1
    tau1 = data[0][0] - data[1][0]
    tau2 = data[2][0] - data[0][0]
    tau3 = data[2][0] - data[1][0]

    a1 = tau3/tau2
    a3 = -1*tau1/tau2

    R1 = data[0][3]
    R2 = data[1][3]
    R3 = data[2][3]

    rhoHat1 = od.rhoFinder(data[0][1], data[0][2], 1)
    rhoHat2 = od.rhoFinder(data[1][1], data[1][2], 1)
    rhoHat3 = od.rhoFinder(data[2][1], data[2][2], 1)

    [D0, D11, D12, D13, D21, D22, D23, D31, D32, D33] = od.DxxFinder(rhoHat1, R1, rhoHat2, R2, rhoHat3, R3)
    bottom = (od.dot(od.cross(rhoHat1, rhoHat2), rhoHat3))

    rho1 = (a1*D11 - D12 + a3*D13)/(a1*bottom)
    rho2 = (a1*D21 - D22 + a3*D23)/(-1*bottom)
    rho3 = (a1*D31 - D32 + a3*D33)/(a3*D0)

    # Initial guesses for r and r_dot (for f and g series)
    r1 = rho1 - R1
    r2 = rho2 - R2
    r3 = rho3 - R3

    # TODO: I think this only works when the three observations are evenly spaced.
    # Supposedly there's a handout that tells how to correct for this
    # r2_dot initial guess:
    r2_dot = (r3-r1)/tau2


    #Iterative Process:
