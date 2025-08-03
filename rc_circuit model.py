"""
RC Circuit Simulation 
-----------------------------------------------------

This script simulates the charging and discharging of a capacitor in an RC circuit.
We analyze both numerically (via ODE solver) and analytically.

Mathematical Derivation:
------------------------

For a capacitor charging in an RC series circuit with a constant voltage source V0:

KVL gives:
    V0 = V_R + V_C = R * i(t) + V_C(t)

But,
    i(t) = C * dV_C/dt

So,
    V0 = R * C * dV_C/dt + V_C(t)

Rearranged:
    dV_C/dt + (1/RC) * V_C = V0 / RC  → (1)

This is a linear first-order nonhomogeneous ODE with solution:
    V_C(t) = V0 * (1 - e^(-t / RC))

For discharging:
    dV_C/dt = -V_C / RC  → (2)

This is a linear homogeneous ODE with solution:
    V_C(t) = V0 * e^(-t / RC)

Where:
    R = Resistance in ohms
    C = Capacitance in farads
    τ = RC = time constant

At t = τ:
    Charging → V_C = 0.63 * V0
    Discharging → V_C = 0.37 * V0

We simulate both scenarios and compare the numerical solution to the analytical one.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
R = 1000           # Resistance (ohms)
C = 0.001          # Capacitance (farads)
V0 = 5             # Input voltage
tau = R * C        # Time constant (seconds)

# Time array (up to 5 time constants)
t = np.linspace(0, 5 * tau, 1000)

# Charging phase ODE
def charging_eq(V, t, R, C, V0):
    return (V0 - V) / (R * C)

# Initial voltage (uncharged capacitor)
V_init = 0

# Solve numerically
V_charge_numerical = odeint(charging_eq, V_init, t, args=(R, C, V0)).flatten()

# Analytical solution
V_charge_analytical = V0 * (1 - np.exp(-t / tau))

# Plot charging
plt.figure(figsize=(10, 5))
plt.plot(t, V_charge_numerical, 'b', label='Numerical (ODE)')
plt.plot(t, V_charge_analytical, 'r--', label='Analytical')
plt.title('RC Circuit: Capacitor Charging')
plt.xlabel('Time (s)')
plt.ylabel('Voltage across Capacitor (V)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Discharging phase ODE
def discharging_eq(V, t, R, C):
    return -V / (R * C)

# Solve numerically from V0
V_discharge_numerical = odeint(discharging_eq, V0, t, args=(R, C)).flatten()

# Analytical solution
V_discharge_analytical = V0 * np.exp(-t / tau)

# Plot discharging
plt.figure(figsize=(10, 5))
plt.plot(t, V_discharge_numerical, 'g', label='Numerical (ODE)')
plt.plot(t, V_discharge_analytical, 'k--', label='Analytical')
plt.title('RC Circuit: Capacitor Discharging')
plt.xlabel('Time (s)')
plt.ylabel('Voltage across Capacitor (V)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
