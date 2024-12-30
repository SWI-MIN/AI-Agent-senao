# inference.py
# 推理
from sentence_transformers import util
import cv2

'''transcribe_audio(語音, 語音轉文字模型)
    - 處理音頻並轉換為文字
    return 轉換後的文字
'''
def transcribe_audio(audio, audio_model):
    result = audio_model.transcribe(audio)
    return result["text"]

'''analyze_text(輸入文字(語音轉的文字), 語意模型, 關鍵詞語意向量)
    - 語意識別推理動作(分析文字)
    return 相似度最高的動作為何, 可信度為多少
'''
def analyze_text(text, semantic_model, action_embeddings):
    text_embedding = semantic_model.encode(text, convert_to_tensor=True)
    
    # 計算相似度
    similarities = {}
    for action, embeddings in action_embeddings.items():
        max_similarity = max([util.pytorch_cos_sim(text_embedding, emb).item() for emb in embeddings])
        similarities[action] = max_similarity
    
    # 找到最相似的動作
    best_match = max(similarities, key=similarities.get)
    confidence = similarities[best_match]
    print("\n匹配動作: ", best_match, "可信度: ", confidence)

    return best_match, confidence

'''people_detection(YOLO模型, 圖像)
    - 檢測人數並儲存圖片與標記結果
'''
def people_detection(YOLO_model, frame):
    # 使用 YOLO 模型進行人數檢測
    results = YOLO_model(frame)
    people_count = 0

    # 分析檢測結果
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])  # 類別 ID
            confidence = box[4]  # 檢測框的信心分數
            
            # 'person' 在 COCO 資料集中的 ID 是 0，且信心大於0.33才算有效檢測(由於筆電內鍵攝像頭很差，信心度調低比較準)
            if class_id == 0 and confidence > 0.33:
                people_count += 1

    print(f"檢測到的人數: {people_count}")

    annotated_frame = results[0].plot()  # 檢測結果
    # 保存檢測結果
    cv2.imwrite('annotated_frame.jpg', annotated_frame)
    print("檢測結果保存為 annotated_frame.jpg")
