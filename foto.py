from pathlib import Path
import time
import cv2

# Neue, zugeschnittene Aufnahmen getrennt von der alten Vollbild-Session speichern.
OUTPUT_DIR = Path("data/banana/session_03/test")

# Zeit zwischen automatischen Bildern in Sekunden
INTERVAL_SECONDS = 0.7

# Muss mit dem mittleren Erkennungsbereich in src/gui/main_window.py übereinstimmen
BOX_WIDTH_RATIO = 0.60
BOX_HEIGHT_RATIO = 0.60

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden.")

automatic_capture = False
last_capture_time = 0
image_number = 0

window_name = "Fruit Vision - Aufnahme"
cv2.namedWindow(window_name)

capture_requested = False


def handle_mouse_click(event, x, y, flags, param):
    global capture_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_requested = True


cv2.setMouseCallback(window_name, handle_mouse_click)

print("Linksklick: einzelnes Bild | [r] automatische Aufnahme an/aus | [s] einzelnes Bild | [q] beenden")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Kein Kamerabild erhalten.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    box_width = int(width * BOX_WIDTH_RATIO)
    box_height = int(height * BOX_HEIGHT_RATIO)
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Dieser Ausschnitt entspricht dem Bildbereich, den das Modell später sieht.
    roi = frame[y1:y2, x1:x2]

    # Rahmen nur in der Vorschau zeichnen, nicht in das gespeicherte Bild.
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow(window_name, preview)

    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    save_image = automatic_capture and now - last_capture_time >= INTERVAL_SECONDS

    if capture_requested:
        save_image = True
        capture_requested = False

    if key == ord("r"):
        automatic_capture = not automatic_capture
        print(f"Automatische Aufnahme: {automatic_capture}")

    elif key == ord("s"):
        save_image = True

    elif key == ord("q"):
        break

    if save_image:
        image_number += 1
        filename = OUTPUT_DIR / f"banana_{image_number:04d}.jpg"
        cv2.imwrite(str(filename), roi)
        last_capture_time = now
        print(f"Gespeichert: {filename}")

camera.release()
cv2.destroyAllWindows()
