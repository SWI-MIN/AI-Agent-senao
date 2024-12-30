# action.py
import cv2
import inference
import pygame
import sounddevice as sd
import wave
import io

# 錄音動作
def record_audio(file_path, duration=5, samplerate=16000):
    print("🎙️ 開始錄音，按 Enter 結束...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    input("結束錄音...")
    sd.wait()
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

# 執行動作
def perform_action(best_match, confidence, YOLO_model, audio_model):
    if confidence > 0.65: # 此處調整語音輸入可信度的靈敏度
        if best_match == "播放聲音":
            print("🔊 執行播放音樂動作")
            
            pygame.mixer.init()# 初始化 pygame 混音器
            pygame.mixer.music.load("./Sound/ErsatzBossa.mp3")# 載入音樂檔案
            pygame.mixer.music.play()# 播放音樂

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                continue
            
            # 停止音樂並釋放資源
            pygame.mixer.music.stop()
            pygame.mixer.quit()

        elif best_match == "語音轉文字":
            print("🎤 語音輸入後將會將此段語音轉文字保存")
            record_audio("./Sound/mic_input.wav")
            audio_text = inference.transcribe_audio("./Sound/mic_input.wav", audio_model)
            print(f"轉換文字: {audio_text}")
            print("💾 保存為 TXT 文件")
            with open("audio_text.txt", "w",encoding="utf-8") as f:
                f.write(audio_text)

        elif best_match == "現場人數":
            print("📸 啟動相機並進行人數識別")
            try:
                cap = cv2.VideoCapture(0) # 開啟攝像頭 (通常 0 是內建攝像頭)
                if not cap.isOpened():
                    raise RuntimeError("無法打開攝像頭")
                # 跳過前 15 幀
                frame_count = 0
                while frame_count < 15:
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("擷取圖片失敗")
                    frame_count += 1
                print("影像擷取成功")
                inference.people_detection(YOLO_model, frame)
            except RuntimeError as e:
                print(f"❌ 錯誤: {e}")
            finally:
                cap.release() # 釋放攝像頭資源

    else:
        print("⚠️ 未找到匹配的動作")

