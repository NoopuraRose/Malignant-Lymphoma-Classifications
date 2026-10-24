from ultralytics import YOLO 

# Defining model
model_name = "runs/classify/train/weights/best.pt"
test_file = "image.tif"

model = YOLO(model_name)

# prediction
result = model(test_file)
data = result[0]
label = data.names[data.probs.top1] 
conf = data.probs.top1conf.item() 
print(f"{label} ({conf:.2%})")