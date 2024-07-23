import rebound
import numpy as np
from matplotlib import pyplot as plt
import os


discard_file_name = "discards.txt" #Rename file
try:
    os.remove(discard_file_name)
except:
    pass

# Collision Detection
def collision_discard_log(sim_pointer, collision, discard_file_name=discard_file_name):
    sim = sim_pointer.contents
    id_p1 = sim.particles[collision.p1].hash.value
    id_p2 = sim.particles[collision.p2].hash.value

    discard_file = open(discard_file_name, "a")

    if id_p1 > id_p2:
        print("Particles {0} collided with {1} at {2} yrs".format(id_p1, id_p2, sim.t))
        print("Particles {0} collided with {1} at {2} yrs".format(id_p1, id_p2, sim.t), file=discard_file)
        print("Removing particle {0}".format(id_p1))
        ToRemove = 1
    else:
        print("Particles {0} collided with {1} at {2} yrs".format(id_p2, id_p1, sim.t))
        print("Particles {0} collided with {1} at {2} yrs".format(id_p2, id_p1, sim.t), file=discard_file)
        print("Removing particle {0}".format(id_p2))
        ToRemove = 2
    discard_file.close()
    return ToRemove


k = 0.0172020989484
sim = rebound.Simulation()
sim.integratior = "mercurius"
sim.units = ("yr", "AU", "Msun")
date = "2024-07-04 13:45"
h = rebound.hash

sim.add("Sun", date=date, hash=0)
sim.particles[h(0)].r = 0.0046524726 #This is the radius of the Sun in AU
sim.add("Venus", hash=1)
sim.particles[h(1)].r = 0.0000404551 #AU
sim.add("Earth", hash=2)
sim.particles[h(2)].r = 0.0000426343 #AU
sim.add("Mars", hash=3)
sim.particles[h(3)].r = 0.0000227009 #AU
sim.add("Jupiter", hash=4)
sim.particles[h(4)].r = 0.0004778945 #AU
sim.add("Saturn", hash=5)
sim.particles[h(5)].r = 0.0004028667 #AU
sim.add("Uranus", hash=6)
sim.particles[h(6)].r = 0.0001708514 #AU
sim.add("Neptune", hash=7)
sim.particles[h(7)].r = 0.0001655371 #AU

N_pl = 8 #Number of planets + 1 (for the sun)


a = 1.89340700183398	
e = 5.380421005600703E-01	
inc = np.deg2rad(4.120840457227408E+01)
Omega = np.deg2rad(6.345424017771230E+01)
omega = np.deg2rad(2.931127831868882E+02)	
M = np.deg2rad(3.097026540393938E+02)

aSTD = 0.26
eSTD = 0.026
iSTD = np.deg2rad(3.9)
OSTD = np.deg2rad(1.7)
oSTD = np.deg2rad(6.3)
mSTD = np.deg2rad(9.2)



# 1866 Sisyphus
# sim.add(x=-0.79851225, y=-1.53418074,  z=0.0250124,
#         vx=0.63856677*k*365, vy=0.13549047*k*365, vz=-0.44786697*k*365,
#         hash=100) # Vector Declaration
sim.add(a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M, hash=100) #Element Declaration
sim.particles[h(100)].r = 2.83426494E-8 #AU

# print("Mass of the Sun = {}".format(sim.particles[h(0)].m))
# print("GM = {}".format(sim.G*sim.particles[h(0)].m))
# print("mos", 4*np.pi**2) #This should equal the one above (it does right now)

# rebound.OrbitPlot(sim,unitlabel="[AU]", color=[ "black", "blue", "red", "brown", "brown", "blue", "blue", "green"])
# rebound.OrbitPlot(sim,unitlabel="[AU]", color=(N_p1-1)*["black"]+["red"])




N_tp=15

x = sim.particles[h(100)].x
y = sim.particles[h(100)].y
z = sim.particles[h(100)].z
vx = sim.particles[h(100)].vx
vy = sim.particles[h(100)].vy
vz = sim.particles[h(100)].vz


for j in range(1, N_tp):
    aNRM = np.random.normal(loc=a, scale=aSTD)
    eNRM = np.random.normal(loc=e, scale=eSTD)
    iNRM = np.random.normal(loc=inc, scale=iSTD)
    ONRM = np.random.normal(loc=Omega, scale=OSTD)
    oNRM = np.random.normal(loc=omega, scale=oSTD)
    mNRM = np.random.normal(loc=M, scale=mSTD)

    sim.add(a=aNRM, e=eNRM, inc=iNRM, Omega=ONRM, omega=oNRM, M=mNRM, hash=101+j)

rebound.OrbitPlot(sim,unitlabel="[AU]", color=(N_pl-1)*["black"] + (N_tp)*["red"])
plt.show()

sim.exit_max_distance = 1000
sim.collision = "direct"
sim.collision_resolve = collision_discard_log
sim.move_to_com
tend = 50e6
tout = 1000
sim.dt = sim.particles[h(1)].P / 25

archive = "archive.bin"
sim.save_to_file(archive, interval=tout, delete_file=True)
times = np.arange(0,tend,tout)
Nsteps = len(times)

for i in range(Nsteps):
    try:
        sim.integrate(times[i], exact_finish_time=0)
    except rebound.Escape as error:
        for j in range(sim.N):
            p = sim.particles[j]
            d2 = p.x**2 + p.y**2 + p.z**2
            if d2>(sim.exit_max_distance**2):
                index=j
        pid = sim.particles[index].hash.value
        print("Particle {0:2d} was too far from the Sun at {1:12.6e} yrs".format(pid, sim.t))
        discard_file = open(discard_file_name, "a")
        print("Particle {0:2d} was too far from the Sun at {1:12.6e} yrs".format(pid, sim.t), file= discard_file)
        discard_file.close()
        sim.remove(index=index)

    print("Time {0:6.3f} Myr-- Fraction Done {1:5.4f} -- # of Clones {2}".format(sim.t/1e6, sim.t/tend, sim.N-N_pl))
    if sim.N <= N_pl:
        print("No more test particles, ending simulation")
        break

# sim.status()