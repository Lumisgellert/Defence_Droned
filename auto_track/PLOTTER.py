# --- oben ---
import matplotlib
matplotlib.use("TkAgg")          # falls Qt fehlt; sonst "Qt5Agg"
import matplotlib.pyplot as plt
from collections import deque

class LivePlot2D:
    def __init__(self, window_s=8.0, fps_hint=60):
        plt.ion()
        self.fig, (self.ax_e, self.ax_u) = plt.subplots(2, 1, sharex=True)
        (self.l_ex,) = self.ax_e.plot([], [], label="dx")
        (self.l_ey,) = self.ax_e.plot([], [], label="dy")
        (self.l_uy,) = self.ax_u.plot([], [], label="yaw")
        (self.l_ut,) = self.ax_u.plot([], [], label="tilt")
        self.ax_e.legend(); self.ax_u.legend()
        self.ax_e.grid(True); self.ax_u.grid(True)
        self.t0 = None; self.win = window_s; self.pause = 1.0/fps_hint
        self.t=deque(); self.ex=deque(); self.ey=deque(); self.uy=deque(); self.ut=deque()
    def update(self, t_abs, dx, dy, uyaw, utilt):
        if self.t0 is None: self.t0 = t_abs
        t = t_abs - self.t0
        self.t.append(t); self.ex.append(dx); self.ey.append(dy); self.uy.append(uyaw); self.ut.append(utilt)
        tmin = t - self.win
        while self.t and self.t[0] < tmin:
            self.t.popleft(); self.ex.popleft(); self.ey.popleft(); self.uy.popleft(); self.ut.popleft()
        self.l_ex.set_data(self.t, self.ex); self.l_ey.set_data(self.t, self.ey)
        self.l_uy.set_data(self.t, self.uy); self.l_ut.set_data(self.t, self.ut)
        if self.t:
            self.ax_e.set_xlim(max(0,self.t[0]), max(2,self.t[-1]))
            def autos(ax, data):
                if not data: return
                lo, hi = min(data), max(data)
                if lo == hi: lo -= 1; hi += 1
                pad = 0.1*(hi-lo); ax.set_ylim(lo-pad, hi+pad)
            autos(self.ax_e, list(self.ex)+list(self.ey))
            autos(self.ax_u, list(self.uy)+list(self.ut))
        self.fig.canvas.draw(); plt.pause(self.pause)

plot = LivePlot2D(window_s=8.0, fps_hint=60)
