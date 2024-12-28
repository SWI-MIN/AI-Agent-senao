
# 執行動作
def perform_action(best_match, confidence, text):
    if confidence > 0.5:
        if best_match == "播放聲音":
            print("🔊 執行播放音樂動作")
            # 加入播放音樂的程式碼
            # 加入控制硬體
        elif best_match == "語音轉文字":
            print("💾 保存為 TXT 文件")
            # with open("output.txt", "w") as f:
            #     f.write(text)

            # 依時間決定要直接保存或者再讓使用者輸入語音後保存
        elif best_match == "現場人數":
            print("📸 啟動相機並進行人數識別")

            # 啟用筆電攝像頭
            # 加入 YOLO 識別人數的程式碼
    else:
        print("⚠️ 未找到匹配的動作")
