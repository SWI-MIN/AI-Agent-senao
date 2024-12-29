import cv2
import inference
from playsound import playsound

# 執行動作
def perform_action(best_match, confidence, text, YOLO_model):
    if confidence > 0.5:
        if best_match == "播放聲音":
            print("🔊 執行播放音樂動作")
            '''控制硬體播放聲音
                - 這套件在DEF裡面時不能有空格或特殊字符，直接執行可以
            '''
            # 這裡考慮添加多線程，發聲同時不影響操作
            playsound("./Sound/ErsatzBossa.mp3")

        elif best_match == "語音轉文字":
            print("💾 保存為 TXT 文件")
            # with open("output.txt", "w") as f:
            #     f.write(text)

            # 依時間決定要直接保存或者再讓使用者輸入語音後保存
        elif best_match == "現場人數":
            print("📸 啟動相機並進行人數識別")

            # 開啟攝像頭 (通常 0 是內建攝像頭)
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
                inference.people_detection(YOLO_model, frame)
            cap.release() # 釋放攝像頭資源

    else:
        print("⚠️ 未找到匹配的動作")


