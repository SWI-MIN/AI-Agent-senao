# init.py
import whisper
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

'''init()
    - 初始化語音轉文字 Whisper 模型
    - 初始化語意識別 paraphrase-multilingual-MiniLM-L12-v2 模型，支援多語言
    - 初始化YOLO v11s 模型，V11n與V11s時間差50ms，準確度差時間差50ms，但準確度更高
    - 建立關鍵詞並將其向量化，後續比對之用
'''
def init():
    
    ''' 加載 Whisper 模型
        - 可以選擇不同大小的模型，由小到大如 tiny, base, small, medium, large 
        - 用base即可(139M)，tiny不准(72M)，small太大(461M)
        - Whisper(base)本地CPU I7 9750H 略慢
    '''
    audio_model = whisper.load_model("base", device="cpu")

    # 初始化語意模型
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 初始化 YOLO 模型
    YOLO_model = YOLO('.\Model\yolo11s.pt')

    # 定義語意關鍵字和動作（支持多關鍵字）
    actions = {
        "播放聲音": ["播放聲音", "播放音樂", "開始播放"],
        "語音轉文字": ["語音轉文字", "STT", "Speech to text", "轉成文字"],
        "現場人數": ["現場人數", "人數檢測", "數一下有多少人"]
    }

    # 將所有關鍵字轉換為語意向量
    action_embeddings = {}
    for action, keywords in actions.items():
        action_embeddings[action] = [semantic_model.encode(keyword, convert_to_tensor=True) for keyword in keywords]

    return audio_model, semantic_model, YOLO_model, action_embeddings