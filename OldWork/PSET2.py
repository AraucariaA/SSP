#PSET2

import numpy as np
from matplotlib import pyplot as plt

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

    #First Quadrent
    if vertsgn >= 0 and horisgn > 0:
        result = abs(angcos)
    
    #Second Quadrent
    elif vertsgn > 0 and horisgn <= 0:
        result = abs(angcos)

    #Third Quadrent
    elif vertsgn <= 0 and horisgn < 0:
        result = abs((np.pi*2)-angcos)

    #Fourth Quadrent
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
    if(arcs == 30):
        print(f'sign: {sign} deg: {deg}, d1: {d1}, d2: {d2}, d3: {d3}')
    return deg

# print(DMStoDeg(6,7,22.8))

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
    if len(vector) ==2:
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


#PS2-8
#a)
def spiral():

    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rmax(2)

    plt.show()
    