"""
Servo Motor Control: Modeling and Simulation
--------------------------------------------

State variables:
    x1 = θ (angular position in radians)
    x2 = ω = dθ/dt (angular velocity in rad/s)
    x3 = i (armature current in A)

Equations:
    dθ/dt = ω
    dω/dt = (-b/J)*ω + (K_t/J)*i
    di/dt = (-K_e/L)*ω - (R/L)*i + (1/L)*V(t)

We simulate step input voltage V(t) and plot θ(t).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
J = 0.01    # moment of inertia (kg·m^2)
b = 0.1     # viscous friction (N·m·s)
K_t = 0.01  # torque constant (N·m/A)
K_e = 0.01  # back EMF constant (V·s/rad)
R = 1       # armature resistance (ohm)
L = 0.5     # armature inductance (H)

# Input voltage (step)
def input_voltage(t):
    return 5.0 if t >= 0 else 0.0  # step from 0 to 5V at t=0

# State derivatives
def servo_motor_deriv(x, t):
    theta, omega, i = x
    V = input_voltage(t)
    
    dtheta_dt = omega
    domega_dt = (-b / J) * omega + (K_t / J) * i
    di_dt = (-K_e / L) * omega - (R / L) * i + (1 / L) * V
    
    return [dtheta_dt, domega_dt, di_dt]

# Initial conditions: motor at rest
x0 = [0, 0, 0]

# Time vector
t = np.linspace(0, 2, 1000)  # simulate for 2 seconds

# Solve ODE
sol = odeint(servo_motor_deriv, x0, t)

theta = sol[:, 0]
omega = sol[:, 1]
current = sol[:, 2]

# Plot results
plt.figure(figsize=(10, 7))

plt.subplot(3,1,1)
plt.plot(t, theta)
plt.title('Servo Motor Response to Step Input Voltage')
plt.ylabel('Position θ (rad)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, omega)
plt.ylabel('Angular Velocity ω (rad/s)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t, current)
plt.xlabel('Time (s)')
plt.ylabel('Armature Current i (A)')
plt.grid(True)

plt.tight_layout()
plt.show()
