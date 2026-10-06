from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\alexa\Desktop\Fruit-Vision\models\trained\Classification_1_own_data\weights\best.pt")

    model.train(
        data = r"C:\Users\alexa\Desktop\Fruit-Vision\trainingsdata\classification\Fruit360",
        epochs = 20,
        imgsz = 100,
        batch = 20,
        patience = 20,
        project =r"C:\Users\alexa\Desktop\Fruit-Vision\models\trained",
        name = "Classification_1_Fruit_360_plus_own_Model"
    )