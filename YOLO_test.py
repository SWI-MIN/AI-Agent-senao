import cv2
from ultralytics import YOLO

# 初始化 YOLO 模型
model = YOLO('.\Model\yolo11n.pt')

# 開啟攝像頭
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法打開攝像頭")
    exit()

# 擷取一張圖片
ret, frame = cap.read()
if not ret:
    print("擷取圖片失敗")
    exit()
else:
    print("影像擷取成功")
    
cap.release() # 釋放攝像頭資源

# 使用 YOLO 模型進行人數檢測
results = model(frame)
people_count = 0

# 分析檢測結果
for result in results:
    for box in result.boxes.data:
        class_id = int(box[5])  # 類別 ID
        if class_id == 0:  # 'person' 在 COCO 資料集中的 ID 是 0
            people_count += 1

print(f"檢測到的人數: {people_count}")

# 檢測結果
annotated_frame = results[0].plot()

# 保存檢測結果
cv2.imwrite('annotated_frame.jpg', annotated_frame)
print("檢測結果保存為 annotated_frame.jpg")

