import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

K, tau = 1.0, 0.8
Kp, Ki, Kd = 2.0, 1.0, 0.05

# PID(s) = Kp + Ki/s + Kd*s
num_pid = [Kd, Kp, Ki]
den_pid = [1, 0]

# Strecke: G(s) = K / (tau s + 1)
num_g = [K]
den_g = [tau, 1]

# offene Übertragungsfunktion: L(s) = PID(s)*G(s)
num_L = np.polymul(num_pid, num_g)
den_L = np.polymul(den_pid, den_g)

# geschlossener Kreis: T(s) = L(s) / (1 + L(s))
num_T = num_L
den_T = np.polyadd(den_L, num_L)

sys_cl = signal.TransferFunction(num_T, den_T)

# Sprungantwort
t, y = signal.step(sys_cl)
plt.plot(t, np.ones_like(t), label="Sollwert")
plt.plot(t, y, label="Ausgang y(t)")
plt.xlabel("t [s]")
plt.ylabel("Amplitude")
plt.title("Sprungantwort PID-Regler")
plt.grid(True)
plt.legend()
plt.show()
