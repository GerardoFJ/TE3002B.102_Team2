from ultralytics import YOLO

model = YOLO("yolo26s.pt")   # nano 

# Entrenar
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    name="señales_trafico",
    batch = -1,
    lr0=0.01,
    device=0 
)
