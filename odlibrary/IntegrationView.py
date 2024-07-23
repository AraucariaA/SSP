import rebound
from rebound import hash as h
from matplotlib import pyplot as plt

sa = rebound.Simulationarchive("archive.bin")




print("Number of snapshots: {}".format(len(sa)))
print("Time of first and last snapshot: {:.1f}, {:.1f}".format(sa.tmin, sa.tmax))

N_p1 = 8
N_tp = 20

sim = sa[0]
rebound.OrbitPlot(sim,unitlabel="[AU]", color=(N_p1-1)*["black"] + (N_tp+1)*["red"])

for i in sim in enumerate(sa):
    print(sim.t/1e6)

sim = sa[-1]
rebound.OrbitPlot(sim, unitlabel="[AU]", color=(N_p1-1)*["black"] + N_tp*["red"])

plt.show()
# sim.status()

# orbits = sim.orbits()
# for orbit in orbits:
#     print(orbit)