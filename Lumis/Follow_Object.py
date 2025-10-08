from collections import defaultdict
import time, math
import cv2
import numpy as np
from ultralytics import YOLO

# -------- Einstellungen --------
MODEL_PATH = "../Modelle/yolo11x.pt"  # dein Pfad
SOURCE = "Car_video.mp4"                        # 0 = Webcam
TRACKER = "botsort.yaml"        # oder "botsort.yaml"
TARGET_NAMES = ["cup"]            # Zielklasse(n)
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 640
DEADZONE_PX = 30                  # Toleranz um die Bildmitte

# -------- Init --------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Quelle nicht geöffnet.")

names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
name_to_idx = {v: k for k, v in names.items()}
target_cls = {name_to_idx[n] for n in TARGET_NAMES if n in name_to_idx}

track_history = defaultdict(list)

def guidance_text(dx, dy, dead=DEADZONE_PX):
    # dx>0 = Ziel rechts von Mitte -> Kamera nach rechts schwenken
    moves = []
    if abs(dx) > dead:
        moves.append("rechts" if dx > 0 else "links")
    if abs(dy) > dead:
        moves.append("unten" if dy > 0 else "oben")
    if not moves:
        return "zentriert"
    return "bewegen: " + " & ".join(moves)

def draw_hud(img, cx0, cy0, dx, dy):
    # Fadenkreuz + Deadzone
    h, w = img.shape[:2]
    cv2.line(img, (cx0-20, cy0), (cx0+20, cy0), (255,255,255), 1)
    cv2.line(img, (cx0, cy0-20), (cx0, cy0+20), (255,255,255), 1)
    cv2.rectangle(img, (cx0-DEADZONE_PX, cy0-DEADZONE_PX),
                       (cx0+DEADZONE_PX, cy0+DEADZONE_PX), (200,200,200), 1)
    # Fehlervektor
    cv2.arrowedLine(img, (cx0, cy0), (int(cx0+dx), int(cy0+dy)), (0,255,255), 2, tipLength=0.25)

try:
    prev_t = time.time()
    while True:
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

        target_found = False
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            # Kandidaten: nur Zielklassen
            candidates = []
            for (x1,y1,x2,y2), c, s in zip(xyxy, cls, conf):
                if c in target_cls:
                    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                    d = math.hypot(cx - cx0, cy - cy0)
                    candidates.append((d, (x1,y1,x2,y2), (cx,cy), c, s))

            if candidates:
                # Nächster zur Mitte
                candidates.sort(key=lambda z: z[0])
                _, (x1,y1,x2,y2), (cx,cy), c, s = candidates[0]
                target_found = True

                # Box + Zentroid
                x1i,y1i,x2i,y2i = map(int, [x1,y1,x2,y2])
                cv2.rectangle(out, (x1i,y1i), (x2i,y2i), (0,255,0), 2)
                cv2.circle(out, (int(cx), int(cy)), 4, (0,255,0), -1)
                cv2.putText(out, f"{names[c]} {s:.2f}", (x1i, max(12, y1i-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Fehler relativ zur Bildmitte
                dx, dy = cx - cx0, cy - cy0
                draw_hud(out, cx0, cy0, dx, dy)
                instr = guidance_text(dx, dy)
                cv2.putText(out, f"Abweichung: dx={dx:.0f}px dy={dy:.0f}px | {instr}",
                            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if not target_found:
            # HUD ohne Ziel
            draw_hud(out, cx0, cy0, 0, 0)
            cv2.putText(out, "Ziel nicht gefunden", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("YOLO Zielhilfe", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
