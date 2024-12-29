import whisper
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

def init():
    ''' 加載 Whisper 模型
        - 可以選擇不同大小的模型，由小到大如 tiny, base, small, medium, large 
        - 用base即可(139M)，tiny不准(72M)，small太大(461M)
    '''
    audio_model = whisper.load_model("base")

    # 初始化語意模型
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 初始化 YOLO 模型
    YOLO_model = YOLO('.\Model\yolo11n.pt')

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