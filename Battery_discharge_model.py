"""
Battery Discharge Model 
======================================================

This script simulates the terminal voltage of a battery over time as it discharges under
constant current. It uses a simplified model involving internal resistance and a 
state-of-charge (SOC)-dependent open circuit voltage (OCV).

------------------------------------------------------
1. Mathematical Formulation
------------------------------------------------------

We consider the following parameters:

    Q      = Battery capacity (Ah)
    I(t)   = Discharge current (A), assumed constant
    SOC(t) = State of charge at time t (0 ≤ SOC ≤ 1)
    V_OC   = Open circuit voltage (V)
    V_bat  = Terminal voltage (V)
    R_int  = Internal resistance (Ω)

------------------------------------------------------
2. State of Charge (SOC)
------------------------------------------------------

SOC decreases linearly with discharge current:

    Continuous form:
        SOC(t) = SOC₀ - (1 / Q) ∫₀ᵗ I(τ) dτ

    Discrete form (numerical):
        SOCₖ₊₁ = SOCₖ - (I * Δt) / (Q * 3600)

Where:
    - Δt is the timestep in seconds
    - 3600 converts hours to seconds

------------------------------------------------------
3. Open Circuit Voltage (OCV)
------------------------------------------------------

A linear approximation of OCV as a function of SOC:

    V_OC(SOC) = a * SOC + b

Where:
    - a and b are empirical constants fitted from battery data

------------------------------------------------------
4. Terminal Voltage (V_bat)
------------------------------------------------------

The voltage at the battery terminals is given by:

    V_bat(t) = V_OC(SOC(t)) - I * R_int

This accounts for the internal resistance drop under load.
"""

import numpy as np
import matplotlib.pyplot as plt

# Battery Parameters
Q = 2.0              # Battery capacity in Ah
R_int = 0.05         # Internal resistance in ohms
I_discharge = 1.0    # Constant discharge current in A
SOC0 = 1.0           # Initial SOC (100%)

# OCV parameters: V_OC(SOC) = a * SOC + b
a = 1.2              # slope (V per unit SOC)
b = 2.5              # intercept (V)

# Simulation Parameters
t_end = 3600         # Total time in seconds (1 hour)
dt = 1               # Time step in seconds
n_steps = int(t_end / dt)

# Initialize arrays
time = np.zeros(n_steps)
soc = np.zeros(n_steps)
voc = np.zeros(n_steps)
vbat = np.zeros(n_steps)

# Initial conditions
soc[0] = SOC0
voc[0] = a * soc[0] + b
vbat[0] = voc[0] - I_discharge * R_int

# Simulation Loop
for k in range(1, n_steps):
    # Time update
    time[k] = time[k-1] + dt

    # SOC update
    soc[k] = soc[k-1] - (I_discharge * dt) / (Q * 3600)
    soc[k] = max(soc[k], 0)  # Limit SOC to non-negative values

    # Open Circuit Voltage
    voc[k] = a * soc[k] + b

    # Terminal Voltage
    vbat[k] = voc[k] - I_discharge * R_int

# Plot: Terminal Voltage and OCV
plt.figure(figsize=(10, 6))
plt.plot(time/60, vbat, label='Terminal Voltage V_bat(t)', color='blue')
plt.plot(time/60, voc, label='Open Circuit Voltage V_OC(SOC)', color='green', linestyle='--')
plt.xlabel('Time (minutes)')
plt.ylabel('Voltage (V)')
plt.title('Battery Discharge: Voltage vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: State of Charge
plt.figure(figsize=(10, 4))
plt.plot(time/60, soc, label='State of Charge SOC(t)', color='orange')
plt.xlabel('Time (minutes)')
plt.ylabel('SOC (0 to 1)')
plt.title('Battery Discharge: SOC vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

