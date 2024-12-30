# AI-Agent-senao

### 硬體與環境

- CPU : I7-9750H
- 周邊硬體IO : 筆電自帶
- 系統 : WIN10
- python版本 : python 3.11.9
- 使用virtualenv創建虛擬空間"AI-Agent"，以指令".\AI-Agent\Scripts\activate"啟動

---

### AI工具
#### openAI Whisper
- 用於語音輸入轉文字本地運行
- pip install git+https://github.com/openai/whisper.git
- Whisper 依賴 ffmpeg(v7.1)
- whisper 未使用gpu版(樹梅派無CUDA)

#### paraphrase-multilingual-MiniLM-L12-v2
- 用於語意識別，此版本支援多語言語意識別，包含中文
- pip install transformers sentence-transformers
    - paraphrase : 語義相似性
    - multilingual：該模型支援多語言
    - MiniLM : Mini Language Model，由 Microsoft 開發的壓縮型Transformer模型，具有較少的參數量。
    - L12 : 代表12層 Transformer 編碼器（Layers）

#### YOLO V11
- pip install ultralytics 裡面包含YOLO模型
- 下載權重後引入使用
- 目前選擇YOLOV 11S，其屬於小模型，雖較V11N稍慢50ms，但準確度更高

---

### AI Agent 
決策流程 : 感知->規劃->行動
核心模塊 : 規劃、記憶、工具、行動

本實作包含了 規劃、工具與行動
規劃、工具 = inference.py，行動 = action.py，Agent = main.py

---

## 操作方法與設計邏輯
```
文字輸入(Q離開，SPK語音輸入)
if Q:
    break
elif SPK:
    語音輸入 -> 語意識別
else:
    語意識別

# 識別結果
if 播放 -> 啟用喇叭
if STT -> 啟用語音輸入 -> 語音轉文字再存成TXT
if 人數識別 -> 啟用相機 -> YOLO V11計算人數
```

