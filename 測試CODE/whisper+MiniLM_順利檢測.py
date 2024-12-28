import whisper
from sentence_transformers import SentenceTransformer, util

# 初始化 Whisper 模型
audio_model = whisper.load_model("base")

# 初始化語意模型
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 定義語意關鍵字和動作（多關鍵字支持）
actions = {
    "播放聲音": ["播放聲音", "播放音樂", "開始播放"],
    "語音轉文字": ["語音轉文字", "STT", "Speech to text", "轉成文字"],
    "現場人數": ["現場人數", "人數檢測", "數一下有多少人"]
}


# 將所有關鍵字轉換為語意向量
action_embeddings = {}
for action, keywords in actions.items():
    action_embeddings[action] = [semantic_model.encode(keyword, convert_to_tensor=True) for keyword in keywords]

# 處理音頻並轉換為文字
def transcribe_audio(file_path):
    result = audio_model.transcribe(file_path)
    return result["text"]

# 語意識別觸發動作
def trigger_action(text):
    text_embedding = semantic_model.encode(text, convert_to_tensor=True)
    
    # 計算每個動作下所有關鍵字的相似度，取最大值作為該動作的匹配分數
    similarities = {}
    for action, embeddings in action_embeddings.items():
        max_similarity = max([util.pytorch_cos_sim(text_embedding, emb).item() for emb in embeddings])
        similarities[action] = max_similarity
    
    # 找到最相似的動作
    best_match = max(similarities, key=similarities.get)
    confidence = similarities[best_match]
    
    print(f"語意匹配: {best_match}, 置信度: {confidence}")
    
    # 設定觸發閾值
    if confidence > 0.5:
        if best_match == "播放聲音":
            print("🔊 執行播放音樂動作")
            # 在這裡加入控制硬體播放的程式碼
        elif best_match == "語音轉文字":
            print("💾 保存為 TXT 文件")
            # with open("output.txt", "w") as f:
            #     f.write(text)
        elif best_match == "現場人數":
            print("📸 啟動相機並進行人數識別")
            # 在這裡加入 YOLO 識別人數的程式碼
    else:
        print("⚠️ 未找到匹配的動作")


# 範例執行流程
audio_text = transcribe_audio(".\Sound\person_morespeak.m4a")
print(f"轉換文字: {audio_text}")
trigger_action(audio_text)
