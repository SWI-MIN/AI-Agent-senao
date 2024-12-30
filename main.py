# main.py
# 承擔Agent職責
import init, inference, action
import warnings

# whisper預設使用FP16(GPU CUDA使用)，這裡使用CPU運算會跳警告，同時自動轉為FP32CPU運算
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

def main():
    # 初始化 Whisper 模型
    audio_model, semantic_model, YOLO_model, action_embeddings = init.init()

    while True:
        user_input = input("\n\n⭐⭐⭐請輸入指令 (輸入 'SPK' 進入語音模式，輸入 'Q' 離開): ").strip().upper()

        if user_input == 'Q':
            print("👋 程序結束，再見！")
            break
        
        if user_input == 'SPK':
            print("🎤 進入語音模式")
            action.record_audio("./Sound/mic_input.wav")
            input_text = inference.transcribe_audio("./Sound/mic_input.wav", audio_model)
            print(f"轉換文字: {input_text}")
        else:
            input_text = user_input
        if not input_text.strip():
            print("⚠️ 輸入不得為空")
            continue
        
        best_match, confidence = inference.analyze_text(input_text, semantic_model, action_embeddings)
        action.perform_action(best_match, confidence, YOLO_model, audio_model)

if __name__ == '__main__':
    main()