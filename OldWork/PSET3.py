#PSET3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import PSET1
import PSET2



#PS3-4

#a)

#Things I need to find:
# -!!!!Max Height - DONE
# -flight duration - DONE
# -Speed upon impact - 
# -!!!!Range (dist from start) - DONE

#!!!!
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

# print("peak", peakFinder(900,88,0,0))
# print("duration", durationFinder(900,88,0,0))
# print("range", rangeFinder(900,88,0,0))


#Trying to animate the projectile with matplotlib:


# #TODO: Get Alan to check!!!!!
# class Animator():
#     def __init__(self, rMag, theta, xZero, yZero):
        
#         self.end = rangeFinder(rMag, theta, xZero, yZero)
#         self.peak = peakFinder(rMag, theta, xZero, yZero)
#         self.interval = (self.end-xZero)/120

#         self.fig, self.ax = plt.subplots()
#         #self.t = np.linspace(xZero, self.end, self.interval) #Spacing of x values
#         self.t = np.linspace(xZero, self.end, 100)

#         self.g = -9.81
#         # v0 = 12

#         self.z = yZero + rMag*(self.t) + (9.81/2) * (pow(self.t,2))

#         projectile = self.ax.plot(self.t[0], self.z[0], c="b", s=5, label=f'rMag = {rMag} m/s')
#         self.ax.set(xlim=[xZero, self.end],
#                     ylim=[(self.peak-yZero)*(-1.2), (self.peak-yZero)*(1.2)],
#                     xlabel='Time (s)', ylabel='Height (meters)')
        
#         self.ax.legend()
        
#     def update(self, frame):
#         x = self.t[:frame]
#         y = self.z[:frame]

#         # update the line plot:
#         self.projectile.set_xdata(self.t[:frame])
#         self.projectile.set_ydata(self.z[:frame])
#         return (self.projectile)

#     def animig(self):
#         ani = animation.FuncAnimation(fig=self.fig, func=Animator.update, frames=40, interval=30)
#         plt.show()
        

# Animator(900, 88, 0, 0)

# # Animator.ani = animation.FuncAnimation(fig=self.fig, func=Animator.update, frames=40, interval=30)
# # plt.show()











# PS3-5

fruits = np.array([["Apple","Banana","Blueberry","Cherry"],
["Coconut","Grapefruit","Kumquat","Mango"],
["Nectarine","Orange","Tangerine","Pomegranate"],
["Lemon","Raspberry","Strawberry","Tomato"]])

#a)
# Extract the bottom right element in one command:
def PS35a(fruits):
    print("a", fruits[3][3])

#b)
# Extract the inner 2x2 square in one command
def PS35b(fruits):
    print("b", fruits[1:3, 1:3])

#c)
# Extract the first and third rows in one command.
def PS35c(fruits):
    print("c", fruits[1:4:2,0:4])

#d)
# Extract the inner 2x2 square flipped vertically and horizontally in one command.
def PS35d(fruits):
    print("d", fruits[-2:-4:-1, -2:-4:-1])

#e)
# Swap the first and fourth columns in a few commands. Hint: make a copy of an array using the
# np.copy() method. (Dr.D. needed 5 lines, but it can be done in 3 lines.)
def PS35e(fruits):
    temp2 = np.copy(fruits[0:4, 3:4])
    fruits[0:4, 3:4] = fruits[0:4, 0:1]
    fruits[0:4, 0:1] = temp2
    print("e", fruits)

#f)
# Replace every element in the array above with the string "SLICED!" in one command.
def PS35f(fruits):
    fruits[0:4,0:4] = "SLICED!"
    print("f", fruits)



#PS3-6

#b)
def positionFinder(latitude, altitude, azimuth, lst):

    alt = np.deg2rad(altitude)
    lat = np.deg2rad(latitude)
    az = np.deg2rad(azimuth)

    term1 = (np.sin(alt)) * (np.sin(lat))
    term2 = (np.cos(alt)) * (np.cos(lat)) * (np.cos(az))

    delta = np.rad2deg(np.arcsin(term1 + term2))

    top = np.sin(alt) - (np.sin(delta)) * (np.sin(lat))
    bottom = (np.cos(delta)) * (np.cos(lat))
    decHA = np.arccos(top/bottom)

    hmsHA = PSET2.RAdecimalToHMS(decHA)

    return [delta, hmsHA]

