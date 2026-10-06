from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=r"C:\Users\alexa\Desktop\Fruit-Vision\trainingsdata\segmentation\converted_fixed\data.yaml",
        epochs=40,
        imgsz=416,
        patience=20,
        project =r"C:\Users\alexa\Desktop\Fruit-Vision\models\trained",
        name = "Segmentation meine Daten"
    )