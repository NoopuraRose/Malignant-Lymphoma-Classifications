from ultralytics import YOLO 

# Defining model and dataset
model_name = "yolov8n-cls.pt"
dataset = "Dataset/"
test_file = "image.tif"

model = YOLO(model_name)

# training
model.train(data = dataset, epochs = 10, imgsz = 224)

# prediction
result = model(test_file)
data = result[0]
label = data.names[data.probs.top1] 
conf = data.probs.top1conf.item() 
print(f"{label} ({conf:.2%})")