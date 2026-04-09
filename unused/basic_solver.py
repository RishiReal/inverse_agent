'''
FINITE DIFFERENCE 1D Heat Equation

Domain x in [-1,1].

Discretize with 201 Points. dx = 1/200.
'''

import numpy as np

#make the grid
N = 201
x = np.linspace(-1,1,N)
dx = 1 / 200

# parameters
alpha = 0.05
dt = 0.00025
r = alpha * dt / dx**2 # stability
print(f"r = {r}") 

# initial condition 
T_ic = np.exp(-(x+0.5)**2 / 0.001) + np.exp(-(x-0.5)**2 / 0.001)

print("T^0:", np.round(T_ic, 4))

T_new = T_ic.copy()

time = 0.01

for i in range(int(round(time / dt))):
    T_prev = T_new.copy()

    T_new[1:-1] = T_prev[1:-1] + r * (T_prev[2:] - 2*T_prev[1:-1] + T_prev[:-2])

    # neumann boundary conditions
    # bounday point equals teh neighbor so the slpoe is 0
    T_new[0] =  0
    T_new[-1] = 0
    T_prev = T_new

print("T^1:", np.round(T_new, 4))
#print("T_other^1:", np.round(T_other_new, 4))

# plot
import matplotlib.pyplot as plt
plt.plot(x, T_ic, label = "T initial condition")
plt.plot(x, T_new, label = "T " + str(time))
plt.legend()
plt.show()