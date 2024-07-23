import MethodofGauss as mog
import odlib as od

results = mog.gaussMeth("/home/proctort/Documents/PSETs/InputsandResults/ProctorInputMOG4.csv", 0)

r2 = results[0]
r2_dot = results[1]

expected = {
    "P_a" : 1.893803002492018E+00,
    "P_e" : 5.380421279820976E-01,
    "P_i" : 0.7192223,
    "P_bigO" : 1.107485,
    "P_w" : 5.115783,
    "P_T" : 2460628.684954771306,
    "P_MA" : 5.404643
}

od.valueReturn(r2, r2_dot, 1, expected)


# od.orbitPlotter()