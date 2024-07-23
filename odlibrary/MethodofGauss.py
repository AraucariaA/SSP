#Gauss' Meth
import numpy as np
import odlib as od



# The input file should be formatted such that it contains one line for every usable night of data with the following info:
# • UT date/time at middle of observation (JD),
# • RA (hh:mm:ss.ss), Dec (dd:mm:ss.s), and
# • Sun vector from JPL Horizons (AU, equatorial cartesian, J2000, apparent
# states)

# E.g. file = []

def gaussMeth(filepath, switch):
    '''
    This function takes as inputs:
        - The filepath to the input file, arranged as a csv, inputted as a string
        - If switch=0, then your time input should be in Julian Days (JD).
            If switch=1, then your time input should in Gregorian Days, in the form "YYYY MM DD"
            If you days are decimalized (i.e. DD.DDDD), switch=1 should still work
    This function outputs:
        - The asteroid's position and velocity vectors for the central observation in
            ecliptic rectangular coordinates, in AU and AU/day
        - The range (i.e. rho) to the asteroid for the central observation, in AU
        - The 6 orbital elements, all angles in degrees with respect to the ecliptic;
            report Mean anomaly for the date July 20, 2024 6:00 UT.
    '''

    k = 0.0172020989484
    epsilon = -0.4090926
    cAUDay = (299792458 * 86400 / 149597870700) #AU/day

    #Parsing (this doesn't all need to be in the with statement, but I find that the organization is nicer this way):
    with open(str(filepath)) as file:
        entireArray = file.readlines()
        # print("entireArray", entireArray)

        preData = []

        for i in range(0, len(entireArray)):
            row = entireArray[i].split("/n")
            row = entireArray[i].split(",")
            # row = entireArray[i].split(" ")
            # row = entireArray[i].split(":")
            
            for j in range(0, len(row)):
                try:
                    row[j] = float(row[j])
                except:
                    row[j] = row[j]
                    
            preData.append(row)
        

        data = [[],[],[]]

        for i in range(0, 3):
            #This is to turn the data from Gregorian to JD
            if switch == 1:
                gregD = preData[i][0].split(" ")
                UTC = preData[i][1].split(":")
                
                gregD = [float(s) for s in gregD]
                UTC = [float(s) for s in UTC]

                JD = od.gregtoJD(gregD[0], gregD[1], gregD[2], UTC[0], UTC[1], UTC[2])
            # This is for if you already have Julian Days
            elif switch == 0:
                preData[i][0] = float(preData[i][0])
                JD = preData[i][0]

            data[i].append(JD)

            #Decimalizing RA:
            preRA = preData[i][2].split(":")
            preRA = [float(s) for s in preRA]

            RAhrs = preRA[0]
            RAmin = preRA[1]
            RAsec = preRA[2]
            RArad = od.HMSconverter(RAhrs, RAmin, RAsec, 1)
            data[i].append(RArad)

            # print("preData", preData)

            #Decimalizing DEC:
            preDEC = preData[i][3].split(":")
            preDEC = [float(s) for s in preDEC]

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

    # print("data", data)

    t1 = data[0][0]
    t2 = data[1][0]
    t3 = data[2][0]

    #Since we haven't started operating in AU and Gaussian days, mu = k^2 =! 1
    tau1 = k*(t1 - t2)
    tau2 = k*(t3 - t1)
    tau3 = k*(t3 - t2)
    # print("tau1", tau1)
    # print("tau2", tau2)
    # print("tau3", tau3)

    a1 = tau3/tau2
    a3 = -1*tau1/tau2
   
    R1 = np.array(data[0][3])
    R2 = np.array(data[1][3])
    R3 = np.array(data[2][3])

    rhoHat1 = np.array(od.rhoFinder(data[0][1], data[0][2], 1))
    rhoHat2 = np.array(od.rhoFinder(data[1][1], data[1][2], 1))
    rhoHat3 = np.array(od.rhoFinder(data[2][1], data[2][2], 1))
    # print("rhoHat1", rhoHat1)
    # print("rhoHat2", rhoHat2)
    # print("rhoHat3", rhoHat3)

    [D0, D11, D12, D13, D21, D22, D23, D31, D32, D33] = od.DxxFinder(rhoHat1, R1, rhoHat2, R2, rhoHat3, R3)
    bottom = (od.dot(od.cross(rhoHat2, rhoHat3), rhoHat1))
    # dot(rho1, cross(rho2, rho3))
    # print("d's", [D0, D11, D12, D13, D21, D22, D23, D31, D32, D33])

    rho1Mag = (a1*D11 - D12 + a3*D13)/(a1*D0)
    rho2Mag = (a1*D21 - D22 + a3*D23)/(-1*D0)
    rho3Mag = (a1*D31 - D32 + a3*D33)/(a3*bottom)
    # print("rho1Mag", rho1Mag)
    # print("rho2Mag", rho2Mag)
    # print("rho3Mag", rho3Mag)

    # Initial guesses for r and r_dot (for f and g series)
    r1 = rho1Mag*rhoHat1 - R1
    r2 = rho2Mag*rhoHat2 - R2
    r3 = rho3Mag*rhoHat3 - R3
    # print("r2", r2)

    # t1 = t1 * (58.13244086)
    # t2 = t2 * (58.13244086)
    # t3 = t3 * (58.13244086)

    v12 = (r2 - r1)/(t2-t1)
    v23 = (r3 - r2)/(t3-t2)
    # print("v23", v23)
    
    r2_dot = ((t3-t2)*v12 + (t2-t1)*v23)/(t3-t1)
    r2_dot = r2_dot *(58.13244086) #Gaussifying this
    # print("r2_dot", od.magFinder(r2_dot))
    
    f1, g1 = od.fgSeriesFinder(tau1, r2, r2_dot)
    f3, g3 = od.fgSeriesFinder(tau3, r2, r2_dot)
    # print("f1, g1", f1, g1)
    # print("f3, g3", f3, g3)

    a1 = g3/(f1*g3 - f3*g1)
    a3 = -1*g1/(f1*g3 - f3*g1)
    # print("a1, a3", a1, a3)
    
    error = 5
    counter = 0
    LOOPON = True
    while error > 1e-12 and LOOPON == True: 

        rho1Mag = (a1*D11 - D12 + a3*D13)/(a1*D0)
        rho2Mag = (a1*D21 - D22 + a3*D23)/(-1*D0)
        rho3Mag = (a1*D31 - D32 + a3*D33)/(a3*bottom)
        # print("rho1Mag", rho1Mag)
        # print("rho2Mag", rho2Mag)
        # print("rho3Mag", rho3Mag)

        #Speed of light correction:
        t1corrected = t1 - rho1Mag/cAUDay
        t2corrected = t2 - rho2Mag/cAUDay
        t3corrected = t3 - rho3Mag/cAUDay
        tau1 = k*(t1corrected - t2corrected)
        tau2 = k*(t3corrected - t1corrected)
        tau3 = k*(t3corrected - t2corrected)
        ###########################

        r1 = rho1Mag*rhoHat1 - R1
        r2 = rho2Mag*rhoHat2 - R2
        r3 = rho3Mag*rhoHat3 - R3

        r2Mag = od.magFinder(r2)
        
        #Now let's recalculate r1, r2, and r3:
        r2 = ((g3*r1) - (g1*r3)) / ((f1*g3) - (f3*g1))
        r2_dot = ((f3*r1) - (f1*r3)) / ((f3*g1) - (f1*g3))
        # r2_dot = r2_dot*(58.13244086)

        f1, g1 = od.fgSeriesFinder(tau1, r2, r2_dot)
        f3, g3 = od.fgSeriesFinder(tau3, r2, r2_dot)

        # try:
        #     f1, g1 = od.fgSeriesFinder(tau1, r2, r2_dot)
        #     f3, g3 = od.fgSeriesFinder(tau3, r2, r2_dot)x
        # except:
        #     rho2 = rho2Mag*rhoHat2  
        #     result = [r2, r2_dot, rho2]
        #     return result

        a1 = g3/(f1*g3 - f3*g1)
        a3 = -1*g1/(f1*g3 - f3*g1)

        #Recalculating error:
        if counter>0:
            error = abs((r2Mag-oldR2D)/r2Mag)
        
        oldR2D = r2Mag
        # print(error)
        counter +=1
    

    #Changing from equitorial to ecliptic:
    eclipticMat = np.array([[1,0,0],
                        [0, np.cos(epsilon), -1*np.sin(epsilon)],
                        [0, np.sin(epsilon), np.cos(epsilon)]])

    r2_Eclip = np.matmul(eclipticMat, r2)
    r2_dot_Eclip = np.matmul(eclipticMat, r2_dot)
    
    rho2 = rho2Mag*rhoHat2
    result = [r2_Eclip, r2_dot_Eclip]

    return result


# print("Results:", gaussMeth("/home/proctort/Documents/PSETs/InputsandResults/ProctorInputMOG1.csv", 1))
# MOG2 is the lockstep document
# print("Results:", gaussMeth("/home/proctort/Documents/PSETs/InputsandResults/ProctorInputMOG2.csv", 0))
# MOG3 is our asteroid
print("Results:", gaussMeth("/home/proctort/Documents/PSETs/InputsandResults/ProctorInputMOG3.csv", 1))

# testlst = [[],[]]
# tst = np.array(testlst)
# print("testlst", testlst)

