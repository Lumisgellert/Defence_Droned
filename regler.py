import numpy as np
import matplotlib.pyplot as plt

# --- Strecke: G(s) = K / (tau s + 1) ---
K, tau = 1.0, 0.8

# --- PID-Parameter ---
Kp, Ki, Kd = 0.5, 1.0, 0.05

# --- Simulation ---
dt = 0.001
T  = 10.0
N  = int(T/dt)
r  = 2.0                      # Sprunghöhe

y  = np.zeros(N)
u  = np.zeros(N)
t  = np.arange(N)*dt

e_int = 0.0
e_prev = 0.0
# optional: Ableitungsfilter
alpha = 0.1
d_prev = 0.0

for k in range(1, N):
    # Fehler
    e = r - y[k-1]

    # Integralanteil (mit einfacher Anti-Windup-Begrenzung)
    e_int = np.clip(e_int + e*dt, -5.0, 5.0)

    # Differenzialanteil mit 1. Ordnung Tiefpass auf e'
    de = (e - e_prev)/dt
    d = alpha*d_prev + (1-alpha)*de

    # PID-Ausgang
    u[k] = Kp*e + Ki*e_int + Kd*d

    # Strecken-DGL: dy/dt = (-y + K*u)/tau  -> Vorwärts-Euler
    y[k] = y[k-1] + dt * ((-y[k-1] + K*u[k]) / tau)

    # Update
    e_prev = e
    d_prev = d

# --- Plot ---
plt.figure()
plt.plot(t, np.full_like(t, r), label="Sollwert")
plt.plot(t, y, label="Ausgang y(t)")
plt.xlabel("t [s]")
plt.ylabel("Amplitude")
plt.title("Sprungantwort mit PID")
plt.legend()
plt.grid(True)
plt.show()
