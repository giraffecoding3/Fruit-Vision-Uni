from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\alexa\Desktop\Fruit-Vision\models\untrained\yolo11n_untrained_object_detection.pt")

    model.train(
        data=r"C:\Users\alexa\Desktop\Fruit-Vision\Fruits-detection\data.yaml",
        epochs=80,
        imgsz=640,
        batch=20,
        patience=20,
        project=r"C:\yolo-runs",
        name="Vergleichsmodel"
    )
