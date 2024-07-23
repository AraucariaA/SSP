import math
import numpy as np
from matplotlib import pyplot as plt


#PS5-4

#PSET-5
import numpy as np


#PS5-4
#WORKS!!!!
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












#Does NOT work
def taylorSin(x, N):
    signVert = np.copysign(1, np.sin(x))
    signHori = np.copysign(1, np.cos(x))
    if x>=np.pi:
        x = signVert*signHori*(x%np.pi)
    x = x%(2*np.pi)
    
    sum = 0
    for i in range(0, N):
        picker = (i)%4
        if picker == 0:
            item = np.sin(x)
        elif picker == 1:
            item = np.cos(x)
        elif picker == 2:
            item = -1*np.sin(x)
        elif picker == 3:
            item = -1*np.cos(x)
        sum += (item/(math.factorial(i)))*(x**i)
        plt.scatter(i, sum)
    plt.show()
    return sum


# print(taylorSin(np.pi,5))