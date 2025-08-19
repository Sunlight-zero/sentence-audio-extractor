import os
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word


def load_suppress_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [int(token) for token in f.read().split()]

def transcribe(audio_path: str, model_path: str, file_path: str):
    device = "cuda"
    compute_type = "float16"
    suppress_tokens = load_suppress_tokens(file_path)
    print(f"加载 Whisper 模型 '{model_path}'...")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print(f"开始转写音频 '{os.path.basename(audio_path)}'...")
    segments, _ = model.transcribe(
        audio_path, 
        language="ja", 
        word_timestamps=True,
        suppress_tokens=suppress_tokens
    )
    all_words = [word for segment in segments for word in segment.words]
    print(f"转写完成，识别出 {len(all_words)} 个词。")

    return all_words

if __name__ == "__main__":
    audio_path = r"D:\program\Python\auto-workflows\stt\stt-test\pjsk-test.wav"
    model_path = r"D:/ACGN/gal/whisper/models/faster-whisper-medium"
    file_path = r"D:\program\Python\auto-workflows\stt\stt-test\prevent-kanji-stt\kanji_tokens.txt"
    all_words = transcribe(audio_path, model_path, file_path)
    print(all_words)
