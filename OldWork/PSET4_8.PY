#PSET - 4
#PS4-8
#Velocity Verlet Algorithm

from matplotlib import pyplot as plt
import numpy as np

#General Setup
G = 10
deltaT = 0.03
time = np.arange(0,80, deltaT)

#Star Setup
starMass = 1290#0**30 #kg

#Planet 1 Setup
p1_X = -90 # km
p1_Y = 0 # km
p1Velo_X = 0 # km/s
p1Velo_Y = 12 # km/s
p1xCoords = []
p1yCoords = []


#Planet 2 Setup
p2_X = 0 # km
p2_Y = 80 # km
p2Velo_X = 4 # km/s
p2Velo_Y = 0 # km/s
p2xCoords = []
p2yCoords = []



for i in range(len(time)):
    #Planet One:
    p1Mag = np.sqrt(pow(p1_X,2)+pow(p1_Y,2))

    #Step 1: Acceleration
    p1Accel_X = (-G*starMass*p1_X)/(pow(p1Mag,3))
    p1Accel_Y = (-G*starMass*p1_Y)/(pow(p1Mag,3))

    #Step 2: Position
    p1_X += p1Velo_X*deltaT + (0.5)*p1Accel_X*(pow(deltaT,2))
    p1_Y += p1Velo_Y*deltaT + (0.5)*p1Accel_Y*(pow(deltaT,2))


    p1Mag = np.sqrt(pow(p1_X,2)+pow(p1_Y,2))


    #Step 3: Change in Acceleration
    p1Accel_X_2 = (-G*starMass*p1_X)/(pow(p1Mag,3))
    p1Accel_Y_2 = (-G*starMass*p1_Y)/(pow(p1Mag,3))

    #Step 4: Change in Velocity
    p1Velo_X += (0.5)*(p1Accel_X + p1Accel_X_2)*deltaT
    p1Velo_Y += (0.5)*(p1Accel_Y + p1Accel_Y_2)*deltaT

    p1xCoords.append(p1_X)
    p1yCoords.append(p1_Y)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################

    #Planet Two:
    p2Mag = np.sqrt(pow(p2_X,2)+pow(p2_Y,2))

    #Step 1: Acceleration
    p2Accel_X = (-G*starMass*p2_X)/(pow(p2Mag,3))
    p2Accel_Y = (-G*starMass*p2_Y)/(pow(p2Mag,3))

    #Step 2: Position
    p2_X += p2Velo_X*deltaT + (0.5)*p2Accel_X*(pow(deltaT,2))
    p2_Y += p2Velo_Y*deltaT + (0.5)*p2Accel_Y*(pow(deltaT,2))


    p2Mag = np.sqrt(pow(p2_X,2)+pow(p2_Y,2))


    #Step 3: Change in Acceleration
    p2Accel_X_2 = (-G*starMass*p2_X)/(pow(p2Mag,3))
    p2Accel_Y_2 = (-G*starMass*p2_Y)/(pow(p2Mag,3))

    #Step 4: Change in Velocity
    p2Velo_X += (0.5)*(p2Accel_X + p2Accel_X_2)*deltaT
    p2Velo_Y += (0.5)*(p2Accel_Y + p2Accel_Y_2)*deltaT

    p2xCoords.append(p2_X)
    p2yCoords.append(p2_Y)

# plt.scatter(p1xCoords, p1yCoords, color = 'red')
# plt.scatter(p2xCoords, p2yCoords, color = 'green')
# plt.scatter(0,0, color = "black")
# plt.show()
fig, ax = plt.subplots()

for i in range(len(p1xCoords)):
    ax.plot(p1xCoords[i], p1yCoords[i],'ro')
    ax.plot(p2xCoords[i], p2yCoords[i], 'bo')
    plt.pause(0.01)

