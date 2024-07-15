import math
import numpy as np
from matplotlib import pyplot as plt




#PSET1

#PS1-13
def HMSconverter(Hours, Minutes, Seconds, switch):
    """
    If the switch is 1 you will get radians,
    If the switch is 0 you will get degrees
    """

    if switch == 0:
        deg1 = 360*Hours/24
        deg2 = 360*Minutes/(24*60)
        deg3 = 360*Seconds/(24*60*60)
        deg = deg1 + deg2 + deg3
        return deg
    
    if switch == 1:
        rad1 = (2*math.pi)*Hours/24
        rad2 = (2*math.pi)*Minutes/(24*60)
        rad3 = (2*math.pi)*Seconds/(24*60*60)
        rad = rad1 + rad2 + rad3
        return rad


#PS1-14-a
def dot(vec1, vec2):
    '''
    Returns the dot product of two 3D vectors
    Input both as lists
    You can do 2D vectors as well, just set the Z component to 0
    '''
    product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    return product

#PS1-14-b
def cross(vec1, vec2):
    '''
    Returns the cross product of two 3D vectors
    '''
    xpart = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    ypart = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    zpart = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    norm = [xpart, ypart, zpart]

    return norm

#PS1-14-c
def tripleProduct(vec1, vec2, vec3):
    '''
    This crosses vectors two and three, then dots the resultant vector with vector one
    '''
    vec4 = cross(vec2,vec3)
    result = dot(vec1, vec4)
    return(result)


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################

#PSET2

#PS2-4
def integrator(function, start, stop, n):
    '''
    n is the number of intervals you want
    '''
    width = abs((stop-start)/n)
    sum = 0
    for i in np.arange(start, stop, width):
        sum += function(i + (width/2)) * width
    
    return sum


#PS2-5
def quadamb(sine, cosine):
    '''
    This is in the unit circle and returns radians.
    Outputs range from 0 to 2*pi
    '''
    vertsgn = np.sign(sine)
    horisgn = np.sign(cosine)

    #Both of these return radians
    #Sin is between pi/2 and -pi/2
    angsin = np.arcsin(sine)
    #Returns from 0 to pi
    angcos = np.arccos(cosine)

    #First Quadrant
    if vertsgn >= 0 and horisgn > 0:
        result = abs(angcos)
    
    #Second Quadrant
    elif vertsgn > 0 and horisgn <= 0:
        result = abs(angcos)

    #Third Quadrant
    elif vertsgn <= 0 and horisgn < 0:
        result = abs((np.pi*2)-angcos)

    #Fourth Quadrant
    elif vertsgn < 0 and horisgn >= 0:
        result = abs((np.pi*2)-angcos)
    
    return result


#PS2-6

#a)
def DMStoDeg(degrees, arcm, arcs):   
    '''
    Turns Degrees, Minutes, Seconds to Degrees
    (This is typically for Declination)
    '''

    d1 = (degrees)
    d2 = (arcm/60) #60 min in a degree
    d3 = arcs/3600 #3600 sec in a degree
    
    #if str(degrees) == "-0"
    sign = np.copysign(1, degrees)

    deg = d1 + sign*d2 + sign*d3

    return deg


#b)
def RAdecimalToHMS(decimal):
    '''
    Turns degrees to Hours, Minutes, Seconds
    (This is typically for RA)
    '''
    #This might be optional depending on what they meant by decimalized.
    preHours1 = decimal/15
    preHours1 = preHours1%24

    #This is the final hours
    hours = np.floor(preHours1)

    #The decimalized minutes plus seconds
    preMin1 = preHours1 - hours

    preMin2 = preMin1*60
    minutes = np.floor(preMin2)

    preSec1 = preMin2 - minutes
    seconds = preSec1*60

    #print(hours, " : ", minutes, " : ", seconds)
    
    return ((hours, minutes, seconds))

#c)
def DECdecimalToDMS(decimal):
    '''
    Turns degrees to Degrees, Minutes, Seconds
    (This is typically for Declination)
    '''
    sign = np.copysign(1, decimal)
    
    if sign > 0:
        decimal = decimal%90
    elif sign <0:
        decimal = decimal%-90

    if decimal<0:
        degrees = np.ceil(decimal)
    elif decimal>0:
        degrees = np.floor(decimal)
    elif decimal ==0:
        return (0,0,0)
    
    preMin1 = abs(decimal-degrees)
    preMin2 = preMin1*60
    Min = np.floor(preMin2)

    preSec1 = abs(preMin2 - Min)
    Sec = preSec1*60
    
    if degrees == 0 and sign == -1:
        degrees = degrees * sign * -1

    return(degrees, Min, Sec)



#d)
#Why is this giving me an error?????
def magFinder(vector):
    '''
    Gives you the magnitude for a 2D or 3D vector when inputed as a list
    '''
    if len(vector) == 3:
        mag = np.sqrt(pow(vector[0],2) + pow(vector[1],2) + pow(vector[2],2))
    elif len(vector) == 2:
        mag = np.sqrt(pow(vector[0],2) + pow(vector[1],2))
    return float(mag)

#PS2-7
#c)
def specialRotate(vec1, alpha, beta):
    '''
    This takes some vector vec1, then:
    - Rotates it around the z-axis by some angle alpha
    - Then rotates it around the x-axis by some angle beta
    Then outputs the resultant vector
    '''
    matrix = np.array([[np.cos(alpha), np.sin(alpha), 0],
                      [-1*np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta), np.sin(beta)],
                      [np.sin(alpha)*np.sin(beta), -1*np.cos(alpha)*np.sin(beta), np.cos(beta)]])
    
    vec2 = np.matmul(matrix, vec1)
    
    print(vec2)
    return vec2


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


#PSET3

def peakFinder(rMag, theta, xZero, yZero):
    '''
    theta should be in degrees.
    rMag should be in m/s
    If you just want the distance, set 0 = xZero = yZero
    '''

    peak = ((pow(np.sin(rMag),2))/(2*9.81)) + yZero
    return peak

def durationFinder(rMag, theta, xZero, yZero):
    '''
    theta should be in degrees.
    rMag should be in m/s
    If you just want the distance, set 0 = xZero = yZero
    '''

    topPart1 = -1*rMag*(np.sin(np.deg2rad(theta)))
    determinant = np.sqrt(pow(topPart1,2) + 2*yZero*9.81)

    result = (topPart1 - determinant)/(-9.81)
    return result

#!!!!
def rangeFinder(rMag, theta, xZero, yZero):
    '''
    theta should be in degrees.
    rMag should be in m/s
    If you just want the distance, set 0 = xZero = yZero
    '''

    tZero = durationFinder(rMag, theta, xZero, yZero)
    range = xZero + rMag*(np.cos(np.deg2rad(theta)))*tZero

    return range


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


#PSET4
#This only has the projectile sim, not the drag version

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


def ProjSim4a(rMag, theta, xZero, yZero):
    '''
    rMag: initial velocity
    theta: angle from the horizontal of velocity vector
    xZero: x-coordinate from which you wish to start
    yZero: y-coordinate from which you wish to start
    '''
    g = -9.81
    fig, ax = plt.subplots()

    end = rangeFinder(rMag, theta, xZero, yZero)
    peak = peakFinder(rMag, theta, xZero, yZero)

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


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


#PSET5

def newtonsMeth(function, deriv, x1, errTol):
    '''
    The function must be differentiable for all real numbers
    If the function has local minimums and maximums this may also fail
    The first two inputs should be lambda functions
    '''

    y1 = function(x1)
    m = deriv(x1)
    
    # This is the x position when y = 0 for the derivative
    try:
        x = -1*(y1/m) + x1
    except:
        return x
    
    try:
        difference = abs(x-x1)
    except:
        return x
        
    if difference<errTol:
        return x1
    else:
        return newtonsMeth(function, deriv, x, errTol)
    

#Taylor's Sin never worked :(


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


#PSET6
#TODO: Add PSET 6 when finished with it.





###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


# Actual OD Code:

# OD ONE:

def gaussifyer(r, r_dot):

    #If we have to read the data, this will have to be deleted

    # AU's
    # r[0] = 3.970631912189709E-01
    # r[1] = -1.225073703123122E+00
    # r[2] = 4.747425159692229E-01

    # # AU's per day
    # r_dot[0] = 1.139883471649287E-02
    # r_dot[1] = 2.679831533677191E-03
    # r_dot[2] = 3.750852804158524E-03

    for i in range(0, len(r_dot)):
        r_dot[i] = r_dot[i] * (58.13244086)
    #Now everything is in AU and Gaussian Days

    return r, r_dot

def angMomentrumFinder(r, r_dot): 
    '''
    Takes the position vector as a list (x, y, z)
    Takes the velocity vector (r_dot) as a list (vx, vy, vz)
    This gives h!!!!!!!!
    '''
    h = cross(r, r_dot)
    return h



###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


# OD TWO AND THREE:

def SMAFinder(r, r_dot):
    '''
    ~ Returns a float a, the Semi-Major Axis
    '''
    #TODO: Get this checked. I'm not certain this is right.
    #This has to be in AU and Gaussian Days
    Mu = 1

    rMag = magFinder(r)
    vSquare = dot(r_dot, r_dot)

    bottom = (2/rMag) - (vSquare/Mu)
    a = 1/bottom

    #TODO: Change!!
    return a

def eccenFinder(r, r_dot):
    '''
    ~ Returns e, the eccentricity of the orbit
    '''
    #TODO: Get this checked. I'm not certain this is right.
    #This has to be in AU and Gaussian Days
    Mu = 1
    a = SMAFinder(r, r_dot)

    inside = (magFinder(angMomentrumFinder(r,r_dot))**2)/(Mu*a)
    e = np.sqrt(1-inside)

    return e

def inclinFinder(r, r_dot):
    '''
    ~ Returns i, the inclination of the orbital plane from the ecliptic
    '''
    h = angMomentrumFinder(r, r_dot)
    i = np.arccos(h[2]/magFinder(h))

    return i

def bigOmegaFinder(r, r_dot):
    '''
    Solves for longitude of ascending node
    ~ Returns a float, big Omega
    '''
    i = inclinFinder(r, r_dot)
    h = angMomentrumFinder(r, r_dot)
    hMag = magFinder(h)
    sine = h[0]/(hMag*np.sin(i))
    cosine = -1*h[1]/(hMag*np.sin(i))

    bigO = quadamb(sine, cosine)
    return bigO



def preOmegaFinder(r, r_dot):
    '''
    Solves for U and V, two values needed to find little Omega
    ~ Returns two floats (U, V)
    '''

    rMag = magFinder(r)
    # rMag = np.linalg.norm(r)
    i = inclinFinder(r,r_dot)
    bigO = bigOmegaFinder(r, r_dot)
    a = SMAFinder(r, r_dot)
    e = eccenFinder(r, r_dot)
    h = magFinder(angMomentrumFinder(r, r_dot))

    #Finding U:
    sineU = r[2]/(rMag*np.sin(i))
    cosineU = (r[0]*np.cos(bigO) + r[1]*np.sin(bigO))/rMag

    U = quadamb(sineU, cosineU)
    print()

    #Finding v:
    quan = a*(1-e**2)/rMag
    cosineV = (1/e)*(quan - 1)
    sineV = quan*(dot(r, r_dot)/(h*e))
    V = quadamb(sineV, cosineV)

    return (U, V)

def littleOmegaFinder(r, r_dot):
    '''
    ~ Returns little Omega, the argument of perihelion
    '''
    U, V = preOmegaFinder(r, r_dot)
    #Getting little Omega:
    littleOmega = (U - V)%(2*np.pi)
    return 1000000000000
    
def anomalyFinder(r, r_dot):
    '''
    Solves for E and M, the eccentric and mean anomolies respectively
    ~ Returns two floats (E, M)
    '''
    a = SMAFinder(r, r_dot)
    e = eccenFinder(r, r_dot)
    U, V = preOmegaFinder(r, r_dot)
    rMag = magFinder(r)

    cosineE = (a*e + rMag*np.cos(V))/a
    sineE = (rMag*np.sin(V))/(a*np.sqrt(1-e**2))
    E = quadamb(sineE, cosineE)
    M = E - e*np.sin(E)

    return E, M

def periPassFinder(r, r_dot):
    '''
    ~ Returns T in Julian Days!!
    '''
    E, M = anomalyFinder(r, r_dot)
    a = SMAFinder(r, r_dot)
    k = 0.0172020989484
    n = k/(a**(3/2))
    # July:
    # obstime = 2458313.500000000
    # August:
    obstime = 24583335 #Julian days

    T = obstime - M/n
    return T



#######################################################################################

r = [3.970631912189709E-01, -1.225073703123122E+00, 4.747425159692229E-01]
r_dot = [1.139883471649287E-02, 2.679831533677191E-03, 3.750852804158524E-03]



r, r_dot = gaussifyer(r,r_dot)



def valueReturn(r, r_dot, expected):
    '''             
    r and r_dot should be lists. Expected should be a dictionary
    '''
    #Testing supplies
    expected = {
        "P_a" : 1.05671892483881,
        "P_e" : 0.3442798363212599,
        "P_i" : 0.439042,
        "P_bigO" : 4.123131,
        "P_w" : 4.459868,
        "P_T" : 2458158.720849720296,
        "P_MA" : 2.450782
    }

    print("EVERYTHING IS IN RADIANS!!\n")

    print("Angular Momentum per Unit Mass")
    print("\tCalculated:", angMomentrumFinder(r, r_dot), "\n")

    print("Semi-Major-Axis (a)")
    print("\tPredicted:", expected["P_a"])
    print("\tCalculated:", SMAFinder(r, r_dot))
    print("\tPercent Error:", (SMAFinder(r, r_dot)-expected["P_a"])/expected["P_a"])

    print("Eccentricity (e)")
    print("\tPredicted:", expected["P_e"])
    print("\tCalculated:", eccenFinder(r, r_dot))
    print("\tPercent Error:", (eccenFinder(r, r_dot)-expected["P_e"])/expected["P_e"])

    print("Inclination (i)")
    print("\tPredicted:", expected["P_i"])
    print("\tPercent Error:", (inclinFinder(r, r_dot)-expected["P_i"])/expected["P_i"])

    print("Big Omega (OM or bigO)")
    print("\tPredicted:", expected["P_bigO"])
    print("\tCalculated:", bigOmegaFinder(r, r_dot))
    print("\tPercent Error:", (bigOmegaFinder(r, r_dot)-expected["P_bigO"])/expected["P_bigO"])

    print("u and nu (U, V)")
    print("\tCalculated:", preOmegaFinder(r, r_dot))
    
    print("Little Omega (w)")
    print("\tPredicted:", expected["P_w"])
    print("\tCalculated:", littleOmegaFinder(r, r_dot))
    print("\tPercent Error:", (littleOmegaFinder(r, r_dot)-expected["P_w"])/expected["P_w"])

    print("Eccentric Anomoly (E)")
    print("\tCalculated:", anomalyFinder(r, r_dot)[0])

    print("Mean Anomoly (M)")
    print("\tPredicted:", expected["P_MA"])
    print("\tCalculated:", anomalyFinder(r, r_dot)[1])
    print("\tPercent Error:", (anomalyFinder(r, r_dot)[1]-expected["P_MA"])/expected["P_MA"])

    print("Time of Perihelion Passage (T)")
    print("\tPredicted:", expected["P_T"])
    print("\tCalculated:", periPassFinder(r, r_dot))
    print("\tPercent Error:", (periPassFinder(r, r_dot)-expected["P_T"])/expected["P_T"])

    print("Hopefully E", newtonAnomFinder(r, r_dot))


# valueReturn(r,r_dot,0)












###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


# OD EPHEMERIS:



#2
def newtonAnomFinder(r, r_dot):
    '''
    ~ Returns the found e to the specified error value (err)
    '''
    e = eccenFinder(r, r_dot)
    temp, M = anomalyFinder(r, r_dot)
    #I might want to change that error?
    Efinal = newtonsMeth(lambda E: M-(E-e*np.sin(E)), lambda E: -1+e*np.cos(E), M, 1e-12)
    return Efinal

#3
def CartesianConvert(r,r_dot):
    '''
    Turns the position vector into Cartesian form (still a vector)
    The x-axis coincides with the perihelion
    ~ Returns the coordinates in a NumPy array with each element in one column
    '''
    a = SMAFinder(r, r_dot)
    e = eccenFinder(r, r_dot)
    # Do I want a different error? Lower?
    E = newtonAnomFinder(r, r_dot)

    xCart = a*np.cos(E) - a*e
    yCart = a*np.sqrt(1-e**2)*np.sin(E)
    zCart = 0


    coords1 = np.array([[[xCart],[yCart],[zCart]]])
    coordsLST = [xCart, yCart, zCart]
    return coords1, coordsLST

#4
def CartToEcliptic(r,r_dot):
    #TODO: Debug!!!!
    coordsARR, coordsLST = CartesianConvert(r, r_dot)
    w = littleOmegaFinder(coordsLST,r_dot)
    i = inclinFinder(r, r_dot)
    bigO = bigOmegaFinder(r, r_dot)
    

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
    result = np.matmul(prod2, coordsARR)
    
    return result

#5
def EclipToEquat(r, r_dot):
    #TODO: Change for the right date... does this mean I need another input? :(
    epsilon = 0.4090926 #radians
    coords = CartToEcliptic(r, r_dot)
    matrix1 = np.array([[1,0,0],
                        [0, np.cos(epsilon), -1*np.sin(epsilon)],
                        [0, np.sin(epsilon), np.cos(epsilon)]])

    result = np.matmul(matrix1, coords)
    return result


#7
def rhoFinder(r, r_dot):
    R = [-6.57371E-01, 7.09258E-01, 3.0743E-01]
    rho = []
    rhoHat = []
    for j in range(0,3):
        item = R[j] + r[j]
        rho.append(item)
    rhoMag = magFinder(rho)

    for i in range(0,3):
        item = rho[i]/rhoMag
        rhoHat.append(item)
    return rho, rhoHat

def RAandDECFinder(r, r_dot):
    rho, rhoHat = rhoFinder(r, r_dot)
    DEC = np.arcsin(rhoHat[2])
    RA = np.arccos((rhoHat[0])/(np.cos(DEC)))

    sinA = np.sin(RA)
    cosA = np.cos(RA)

    A = quadamb(sinA, cosA)

    return A





# Test Cases:
# print("3", r)
# r, temp = CartesianConvert(r, r_dot)
# print("4", r)
# r = CartToEcliptic(r,r_dot)
# print("5", r)
# r = EclipToEquat(r, r_dot)
# print("6", r)

# print(rhoFinder(r, r_dot))
# print(RAandDECFinder(r, r_dot))