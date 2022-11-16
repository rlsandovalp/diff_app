import numpy as np
import math

def model(Cres, Ccap, D, t, Z):
    C = np.zeros(len(Z))
    for pos_z, z in enumerate(Z):
        C[z] = (1-math.erf(z/(2*math.sqrt(D*t))))*(Cres-Ccap)+Ccap
    return C


Cres = 10
Ccap = 1
Z = np.linspace(0,50,51)
D = 0.002
t = 30

C = model(Cres, Ccap, D, t, Z)
