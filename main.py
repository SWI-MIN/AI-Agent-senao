import whisper
from sentence_transformers import SentenceTransformer, util

import init, inference, action


# 初始化 Whisper 模型
audio_model, semantic_model, action_embeddings = init.init()

# 這裡考慮用無線迴圈接輸入文字，若輸入為SPK(speak)(或者用關鍵詞喚醒否則為文字輸入)，則進入語音模式

# 範例執行流程
audio_text = inference.transcribe_audio(".\Sound\person_morespeak.m4a", audio_model)
print(f"轉換文字: {audio_text}")
best_match, confidence = inference.analyze_text(audio_text, semantic_model, action_embeddings)
action.perform_action(best_match, confidence, audio_text)