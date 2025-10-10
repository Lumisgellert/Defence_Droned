import math

class PID:
    """
    Diskreter P/PI/PID-Regler.
    - Modus über Ki/Kd wählbar:  P  => Ki=0,Kd=0;  PI => Kd=0;  PID => Ki>0,Kd>0
    - Ableitung auf Messung (rauschärmer)
    - 1. Ordnung-Filter auf D-Anteil: tau_d_filt [s]
    - Anti-Windup: 'clamp' oder 'backcalc' (Kaw in 1/s)
    """

    def __init__(self,
                 Kp=1.0, Ki=0.0, Kd=0.0,
                 dt=0.01,
                 umin=-math.inf, umax=math.inf,
                 tau_d_filt=0.01,         # D-Filterzeitkonstante
                 anti_windup='clamp',     # 'clamp' oder 'backcalc'
                 Kaw=1.0):                # nur für backcalc
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.dt = float(dt)
        self.umin, self.umax = float(umin), float(umax)
        self.tau_d_filt = max(0.0, float(tau_d_filt))
        self.anti_windup = anti_windup
        self.Kaw = float(Kaw)

        # Zustände
        self.i = 0.0            # Integrator
        self.y_prev = 0.0       # letzte Messung
        self.d_filt = 0.0       # gefilterter D-Eingang (auf Messung)
        self.u = 0.0

        # Vorabfaktoren
        self._alpha_d = self.dt / (self.tau_d_filt + self.dt) if self.tau_d_filt > 0 else 1.0

    # --- Hauptupdate: r = Sollwert, y = Messwert ---
    def update(self, r, y):
        e = r - y

        # P
        up = self.Kp * e

        # I (vorläufig)
        ui = self.i + self.Ki * e * self.dt

        # D auf Messung: d/dt(-y). Gefiltert.
        dy = (y - self.y_prev) / self.dt
        self.d_filt += self._alpha_d * ((-dy) - self.d_filt)
        ud = self.Kd * self.d_filt

        # Roh-Ausgang
        u_unsat = up + ui + ud

        # Sättigung
        u_sat = max(self.umin, min(self.umax, u_unsat))

        # Anti-Windup
        if self.anti_windup == 'clamp':
            # integriere nur, wenn nicht hart in Sättigung in Fehler-Richtung
            will_increase = (self.Ki * e) > 0
            at_upper = u_sat >= self.umax
            at_lower = u_sat <= self.umin
            if (at_upper and will_increase) or (at_lower and not will_increase):
                pass  # Integrator nicht updaten
            else:
                self.i = ui
        else:  # backcalc
            self.i = ui + self.Kaw * (u_sat - u_unsat) * self.dt

        # Zustände updaten
        self.y_prev = y
        self.u = u_sat
        return u_sat

    # --- Hilfen ---
    def reset(self, x=0.0, y0=None):
        self.i = 0.0
        self.d_filt = 0.0
        if y0 is not None:
            self.y_prev = float(y0)
        self.u = float(x)

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None: self.Kp = float(Kp)
        if Ki is not None: self.Ki = float(Ki)
        if Kd is not None: self.Kd = float(Kd)

    def set_limits(self, umin=None, umax=None):
        if umin is not None: self.umin = float(umin)
        if umax is not None: self.umax = float(umax)

    def set_dt(self, dt):
        self.dt = float(dt)
        self._alpha_d = self.dt / (self.tau_d_filt + self.dt) if self.tau_d_filt > 0 else 1.0
