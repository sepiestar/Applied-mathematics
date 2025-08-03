"""
SIRS Disease Model Simulation
-----------------------------

This script simulates the SIRS (Susceptible-Infectious-Recovered-Susceptible) model,
a compartmental model used in epidemiology to describe the spread of infectious diseases
where recovered individuals can lose immunity and become susceptible again.

Mathematical Model:
-------------------

Let:
    S(t) = number of susceptible individuals at time t
    I(t) = number of infectious individuals at time t
    R(t) = number of recovered individuals at time t
    N    = total population (assumed constant)

Parameters:
    β (beta)   = infection rate (probability of transmission per contact per unit time)
    γ (gamma)  = recovery rate (rate at which infected recover)
    ξ (xi)     = immunity loss rate (rate at which recovered become susceptible again)

The model is governed by the following system of ODEs:

    dS/dt = -β * S * I / N + ξ * R
    dI/dt =  β * S * I / N - γ * I
    dR/dt =  γ * I - ξ * R

This code solves the system using `scipy.integrate.odeint` and plots the result using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Total population, N
N = 1000

# Initial number of infected and recovered individuals
I0 = 10
R0 = 0

# Everyone else is susceptible to infection initially
S0 = N - I0 - R0

# Contact rate (beta), recovery rate (gamma), and immunity loss rate (xi)
beta = 0.3    # transmission rate
gamma = 0.1   # recovery rate
xi = 0.05     # immunity loss rate (R -> S)

# Time grid (in days)
t = np.linspace(0, 160, 160)

# SIRS model differential equations
def deriv(y, t, N, beta, gamma, xi):
    S, I, R = y
    dSdt = -beta * S * I / N + xi * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - xi * R
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIRS equations over the time grid, t
ret = odeint(deriv, y0, t, args=(N, beta, gamma, xi))
S, I, R = ret.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIRS Disease Model Simulation')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
