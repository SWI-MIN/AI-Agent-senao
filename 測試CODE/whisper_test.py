import whisper

''' 加載 Whisper 模型
    - 可以選擇不同大小的模型，由小到大如 tiny, base, small, medium, large 
    - 用base即可(139M)，tiny不准(72M)，small太大(461M)
'''
model = whisper.load_model("base")

# 加載音頻檔案 
audio = whisper.load_audio(".\Sound\STT.m4a") 
audio = whisper.pad_or_trim(audio) 

# 識別語音並將其轉換為文字 
result = model.transcribe(audio) # (audio, language="zh") 可指定語言為中文

# 輸出結果 
print(result["text"])