import numpy as np
import rebound
from rebound import hash as h
from matplotlib import pyplot as plt
import os

# print(os.listdir("."))

sa = rebound.Simulationarchive("archive.bin")
t = np.zeros(len(sa))
a = np.zeros(len(sa))
e = np.zeros(len(sa))

pid = 100

for i, sim in enumerate(sa):
    t[i] = sim.t/1e6
    try:
        a[i] = sim.particles[h(pid)].a
        e[i] = sim.particles[h(pid)].e
    except rebound.ParticleNotFound:
        a = a[:i]
        e = e[:i]
        t = t[:i]
        break

q = a*(1-e)
Q = a*(1+e)

plt.plot(t,a,label="a")
plt.plot(t,q,label="q")
plt.plot(t,Q,label="Q")
plt.yscale("log")
plt.xlabel("time (Million Years)")
plt.ylabel("a (AU)")
plt.title("Particle {0}".format(pid))
plt.legend()