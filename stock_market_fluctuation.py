"""
Geometric Brownian Motion (GBM) Stock Price Simulation
------------------------------------------------------

Mathematical Model:
-------------------

The stock price \( S(t) \) follows the stochastic differential equation (SDE):

    dS(t) = μ S(t) dt + σ S(t) dW(t)

Where:
    - μ (mu) is the expected return (drift coefficient)
    - σ (sigma) is the volatility (diffusion coefficient)
    - W(t) is a standard Wiener process (Brownian motion)

Analytical solution:

    S(t) = S0 * exp((μ - 0.5 * σ²) * t + σ * W(t))

Simulation Approach:
--------------------

We discretize time and simulate increments of Brownian motion \( \Delta W \sim \mathcal{N}(0, \sqrt{\Delta t}) \),
and compute \( S(t) \) iteratively.

"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100        # Initial stock price
mu = 0.05       # Expected annual return (5%)
sigma = 0.2     # Volatility (20% annual std dev)
T = 1.0         # Time horizon in years
dt = 1/252      # Time step (daily, assuming 252 trading days/year)
N = int(T/dt)   # Number of time steps

# Time array
t = np.linspace(0, T, N)

# Generate random Brownian increments
np.random.seed(42)  # for reproducibility
dW = np.random.normal(0, np.sqrt(dt), size=N-1)

# Initialize stock price array
S = np.zeros(N)
S[0] = S0

# Simulate GBM path
for i in range(1, N):
    S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[i-1])

# Plot the simulated stock price path
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Simulated Stock Price')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Geometric Brownian Motion Stock Price Simulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
