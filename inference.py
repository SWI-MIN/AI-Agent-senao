# 推理
from sentence_transformers import util

# 處理音頻並轉換為文字
def transcribe_audio(file_path, audio_model):
    result = audio_model.transcribe(file_path)
    return result["text"]

# 語意識別推理動作(分析文字)
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
    
    return best_match, confidence
