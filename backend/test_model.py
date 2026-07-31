from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="C:\\Users\\YousufTraders\\Downloads\\Skin\\backend\\uploads\\acne-skin-condition_webp.rf.yF0uw7BtOOiQBpiV0nTm (1).webp",   # replace with the actual filename
    conf=0.10,
    save=True
)

print(results[0].boxes)