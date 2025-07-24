import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from faster_whisper import WhisperModel


model_size = "medium" # 或者 "medium.fp16" 来使用半精度
model_path = "D:/ACGN/gal/whisper/models/faster-whisper-medium"

# 在GPU上用FP16半精度加载模型，速度更快，内存占用更小
model = WhisperModel(model_path, device="cuda", compute_type="float16")

print("Starting transcription with word-level timestamps...")

# Set word_timestamps=True to get timestamps for each word
segments, info = model.transcribe(
    r"D:\program\Python\auto-workflows\stt\stt-test\pjsk-test.mp4",
    beam_size=5,
    language="ja",
    initial_prompt="これからはゲームの中で、一歌（いちか）・穂波（ほなみ）・咲希（さき）・志歩（しほ）4人の対話です",
    word_timestamps=True  # <-- The key parameter is here!
)

# Now, iterate through the segments and then the words within each segment
for segment in segments:
    print(f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    for word in segment.words:
        # The 'word' object contains the word itself, and its start and end times
        print(f"  -> Word: [{word.start:.2f}s -> {word.end:.2f}s] '{word.word}' (confidence: {word.probability:.2f})")

# Note: The standard SRT format does not support word-level timestamps.
# You would typically save this detailed information in other formats like JSON,
# or use it directly in your application.
# If you still want an SRT file for general subtitles, you can create it as before.

with open("output_word_level.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        for word in segment.words:
            f.write(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}\n")

print("\nWord-level timestamp details saved to output_word_level.txt")
