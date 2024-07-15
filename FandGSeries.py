
import numpy as np
import odlib as od



def fgSeriesFinder(tau,r2,r2dot):
    r2Mag = od.magFinder(r2)

    #F Series
    preF1 = 1 - ((tau**2)/(2*(r2Mag**3)))
    preF2 = (od.dot(r2, r2dot)*(tau**3))/(2*r2Mag**5)
    preF3 = (tau**4)/(24*(r2Mag**3))
    preF4 = 3*((od.dot(r2dot, r2dot)/(r2Mag**2)) - (1/(r2Mag**3)))
    preF5 = -15*((od.dot(r2dot, r2dot))/(r2Mag**2))**2 + (1/(r2Mag**3))

    f = preF1 + preF2 + preF3*(preF4 + preF5)
    print("f", f)

    #G Series
    preG1 = tau - (tau**3)/(6*(r2Mag**3))
    preG2 = (od.dot(r2, r2dot)*(tau**4))/(4*(r2Mag**5))

    g = preG1 + preG2
    print("g", g)


#Testing Supplies
tau1 = -0.3261857571141891
tau3 = 0.05084081855693949
r2 = [0.26662393644794813, -1.381475976476564, -0.5048589337503169]
r2dot = [0.8442117090940343, -0.39728396707075087, 0.14202728258915864]

fgSeriesFinder(tau3, r2, r2dot)