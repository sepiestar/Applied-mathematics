"""
SEIR Epidemic Model – Mathematical Simulation
=============================================

This script numerically simulates the SEIR model using a system of ODEs.

Model Equations:
----------------
Let:
    S(t) = number of susceptible individuals
    E(t) = number of exposed individuals (infected but not yet infectious)
    I(t) = number of infectious individuals
    R(t) = number of recovered individuals
    N = total population = S + E + I + R

Parameters:
    beta   = transmission rate
    sigma  = rate at which exposed become infectious (1/incubation period)
    gamma  = recovery rate (1/infectious period)

The SEIR model is defined by:

    dS/dt = -beta * S * I / N
    dE/dt = beta * S * I / N - sigma * E
    dI/dt = sigma * E - gamma * I
    dR/dt = gamma * I
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Total population
N = 10000

# Initial number of infected, exposed, and recovered
I0 = 10
E0 = 20
R0 = 0
S0 = N - I0 - E0 - R0  # susceptible initially

# Contact rate, incubation rate, and recovery rate
beta = 0.3       # infection rate per contact
sigma = 1/5.0    # incubation period = 5 days
gamma = 1/7.0    # recovery period = 7 days

# Time grid (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, E0, I0, R0

# The SEIR model differential equations
def deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Integrate the SEIR equations
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.title('SEIR Epidemic Model Simulation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
