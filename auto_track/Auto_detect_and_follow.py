from collections import defaultdict
import time, math, cv2, numpy as np
from ultralytics import YOLO
import Globale_Variable as gv
import threading as th
from printdxy import Printer
from PLOTTER import LivePlot2D
from steuerung_servo import track_step

# -------- Einstellungen --------
MODEL_PATH = "yolo11n.pt"
SOURCE =  0 #f"D:\PyCharm\Drohnen_Projekt\Car_video.mp4"  # oder "Car_video.mp4", 0 für webcam,  "Drohnen_Video.mp4"
TRACKER = "bytetrack.yaml"
TARGET_NAMES = []                # [] = alle Klassen
CONF_THRES = 0.05
IOU_THRES = 0.45
IMG_SIZE = 640
DEADZONE_PX = 30
CLICK_SELECT_RADIUS = 80
ZOOM_SIZE = 150                  # Größe des Zoomfensters in Pixeln
PLOT = False

# Farben
CLR_GRAY = (160,160,160)
CLR_RED  = (0,0,255)
CLR_GRN  = (0,255,0)
CLR_WHT  = (255,255,255)
CLR_YEL  = (0,255,255)

# -------- Init --------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Quelle nicht geöffnet.")

names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
name_to_idx = {v: k for k, v in names.items()}
target_cls = None if not TARGET_NAMES else {name_to_idx[n] for n in TARGET_NAMES if n in name_to_idx}

track_history = defaultdict(list)
selected_id = None
last_click = None          # (x, y)
mouse_pos  = None          # (x, y)

def guidance_text(dx, dy, dead=DEADZONE_PX):
    moves = []
    if abs(dx) > dead: moves.append("rechts" if dx > 0 else "links")
    if abs(dy) > dead: moves.append("unten" if dy > 0 else "oben")
    return "zentriert" if not moves else "bewegen: " + " & ".join(moves)

def draw_hud(img, cx0, cy0, dx, dy):
    h, w = img.shape[:2]
    cv2.line(img, (cx0-20, cy0), (cx0+20, cy0), CLR_WHT, 1)
    cv2.line(img, (cx0, cy0-20), (cx0, cy0+20), CLR_WHT, 1)
    cv2.rectangle(img, (cx0-DEADZONE_PX, cy0-DEADZONE_PX),
                       (cx0+DEADZONE_PX, cy0+DEADZONE_PX), (200,200,200), 1)
    cv2.arrowedLine(img, (cx0, cy0), (int(cx0+dx), int(cy0+dy)), CLR_YEL, 2, tipLength=0.25)

def on_mouse(event, x, y, flags, param):
    global selected_id, last_click, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_id = None
        last_click = None

cv2.namedWindow("YOLO Zielhilfe")
cv2.setMouseCallback("YOLO Zielhilfe", on_mouse)

if PLOT:
    plot = LivePlot2D(window_s=8.0, fps_hint=60)  # ADD
    t_abs = time.perf_counter()                   # ADD


try:
    prev_t = time.time()
    while True:
        #th.Thread(target=Printer, daemon=True).start()  # lässt die funktion Printer aus printdxy parallel laufen
        yaw, tilt = track_step(ex=gv.dx, ey=gv.dy)
        print(yaw, tilt)
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        cx0, cy0 = w//2, h//2

        results = model.track(
            frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
            tracker=TRACKER, persist=True, verbose=False
        )[0]

        out = frame.copy()
        boxes = getattr(results, "boxes", None)
        candidates = []  # (tid, (x1,y1,x2,y2), (cx,cy), cls, conf)

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids  = boxes.id
            ids  = ids.int().cpu().numpy() if ids is not None else np.array([-1]*len(xyxy))

            for (x1,y1,x2,y2), c, s, tid in zip(xyxy, cls, conf, ids):
                if ((target_cls is None) or (c in target_cls)) and tid != -1:
                    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                    candidates.append((int(tid), (x1,y1,x2,y2), (cx,cy), c, s))

        # Hover-Kandidat bestimmen
        hover_tid = None
        if mouse_pos and candidates:
            mx, my = mouse_pos
            best_d = 1e9
            for tid, bb, (cx,cy), c, s in candidates:
                d = math.hypot(cx - mx, cy - my)
                if d < best_d:
                    best_d, hover_tid = d, tid
            if best_d > CLICK_SELECT_RADIUS:
                hover_tid = None

        # Klickauswahl
        if last_click and candidates:
            lx, ly = last_click
            best, best_d = None, 1e9
            for tid, bb, (cx,cy), c, s in candidates:
                d = math.hypot(cx - lx, cy - ly)
                if d < best_d:
                    best_d, best = d, (tid, bb, (cx,cy), c, s)
            if best and best_d <= CLICK_SELECT_RADIUS:
                selected_id = best[0]
            last_click = None

        # HUD
        dx = dy = 0
        draw_hud(out, cx0, cy0, dx, dy)

        # Maus-Radius anzeigen
        if mouse_pos:
            cv2.circle(out, mouse_pos, CLICK_SELECT_RADIUS, (120,120,120), 1)

        # Alle anklickbaren Kandidaten grau umranden, Hover rot hervorheben
        for tid, (x1,y1,x2,y2), (cx,cy), c, s in candidates:
            x1i,y1i,x2i,y2i = map(int, [x1,y1,x2,y2])
            color = CLR_RED if hover_tid == tid else CLR_GRAY
            thickness = 2 if hover_tid == tid else 1
            cv2.rectangle(out, (x1i,y1i), (x2i,y2i), color, thickness)
            # optional kleine Punkte am Zentrum
            cv2.circle(out, (int(cx), int(cy)), 2, color, -1)

        zoom_window = None

        # Ausgewähltes Objekt grün zeichnen + Zoom
        if selected_id is not None:
            sel = None
            for tid, bb, cc, c, s in candidates:
                if tid == selected_id:
                    sel = (tid, bb, cc, c, s); break

            if sel is not None:
                tid, (x1,y1,x2,y2), (cx,cy), c, s = sel
                x1i,y1i,x2i,y2i = map(int, [x1,y1,x2,y2])
                cv2.rectangle(out, (x1i,y1i), (x2i,y2i), CLR_GRN, 2)
                cv2.circle(out, (int(cx), int(cy)), 5, CLR_GRN, -1)
                cv2.putText(out, f"{names[c]} id:{tid} {s:.2f}", (x1i, max(12, y1i-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHT, 1)

                dx, dy = cx - cx0, cy - cy0
                gv.dx = dx
                gv.dy = dy

                draw_hud(out, cx0, cy0, dx, dy)
                instr = guidance_text(dx, dy)
                cv2.putText(out, f"dx={dx:.0f}px dy={dy:.0f}px | {instr}",
                            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_YEL, 2)

                crop = frame[max(0,y1i):min(h,y2i), max(0,x1i):min(w,x2i)]
                if crop.size > 0:
                    zoom_window = cv2.resize(crop, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_CUBIC)
            else:
                cv2.putText(out, "Ziel verloren – Rechtsklick zum Zurücksetzen",
                            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(out, "Linksklick: Objekt wählen | Rechtsklick: löschen",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHT, 2)

        if PLOT:
            # --- Live-Plot updaten (ganz am Ende des Schleifen-Durchlaufs) ---
            t_abs += 0 if 'dt' not in locals() else 0  # ignorieren, nur Platzhalter
            plot.update(time.perf_counter(), dx, dy, yaw, tilt)  # ADD
            # ---------------------------------------------------------------

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHT, 2)

        # Zoomfenster einblenden
        if zoom_window is not None:
            zh, zw = zoom_window.shape[:2]
            out[5:5+zh, 5:5+zw] = zoom_window
            cv2.rectangle(out, (4, 4), (5+zw, 5+zh), CLR_WHT, 1)

        cv2.imshow("YOLO Zielhilfe", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            selected_id = None
            last_click = None

finally:
    cap.release()
    cv2.destroyAllWindows()

