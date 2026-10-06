import cv2

class Camera():
    
    def __init__(self) -> None:

        self.kamera = cv2.VideoCapture(0)
        if not self.kamera.isOpened():
            raise RuntimeError("Bild Fehler")

    def get_picture(self):

        ok, bild = self.kamera.read()
        if not ok:
            raise RuntimeError("Bild kann nicht gelesen werden") 

        return bild

    def close_kamera(self):

        if not self.kamera.isOpened():
            raise RuntimeError("Keine Kamera mehr offen!")
        self.kamera.release()

