from REGLER import PID
yaw_ctl  = PID(Kp=0.05, Ki=0.01, Kd=0.001, dt=0.001)
tilt_ctl = PID(Kp=0.05, Ki=0.01, Kd=0.001, dt=0.001)

def track_step(ex=0, ey=0):
    # PID: r=0 bedeutet "Fehler auf 0 regeln"
    u_yaw  = yaw_ctl.update(0.0, ex)   # Regler-Ausgang für Yaw
    u_tilt = tilt_ctl.update(0.0, ey)  # Regler-Ausgang für Tilt

    # Ausgang an Servos (Stellbefehl):
    pwm_yaw  = int(0 + 400 * u_yaw)
    pwm_tilt = int(0 + 400 * u_tilt)
    return pwm_yaw, pwm_tilt
