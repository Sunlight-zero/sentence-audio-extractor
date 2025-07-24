# -*- coding: utf-8 -*-
"""
代码效果：指定一个游戏的视频或音频文件，和一句日语，从视频中截取出这句话对应的语音。
主要核心工具：
- pykakasi: 将日语文本转换为全平假名
- Demucs: 分离人声和 BGM
- Whisper: 日语语音识别，转换出文字
- fuzz: 字符串模糊匹配模块（解决日语的「表記揺れ」问题）
"""
# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import sys
import shutil
from typing import List, Dict, Optional, Any

# --- 核心依赖 ---
# 请确保已安装所有必要的库。在终端中运行以下命令：
# pip install faster-whisper pydub "thefuzz[speedup]" pykakasi mojimoji demucs

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word
from pydub import AudioSegment
from thefuzz import fuzz
import pykakasi
import mojimoji

# ==============================================================================
# 1. 文本标准化模块
# ==============================================================================

def normalize_japanese_text(text: str) -> str:
    """
    将输入的日文文本标准化为全平假名、半角字符且无标点的形式。

    :param text: 原始日文文本。
    :return: 标准化后的纯平假名字符串。
    """
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    hiragana_text = "".join([item['hira'] for item in result])
    normalized_text = mojimoji.zen_to_han(hiragana_text, kana=False)
    cleaned_text = re.sub(r'[^ぁ-ん]', '', normalized_text)
    return cleaned_text

# ==============================================================================
# 2. 音源分离模块
# ==============================================================================

def separate_vocals(audio_path: str, output_dir: str = "temp_separated", models_path: Optional[str] = None) -> Optional[str]:
    """
    使用 Demucs 将音频文件中的人声和背景音乐分离。

    :param audio_path: 输入的音频文件路径。
    :param output_dir: 分离后文件的输出目录。
    :param models_path: Demucs 模型文件的本地路径。
    :return: 分离出的人声音轨文件路径，如果失败则返回 None。
    """
    print("\n--- 音源分离模块 (引擎: Demucs) ---")
    print("正在分离人声和背景音乐，此过程可能非常耗时，请耐心等待...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    command = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs", "-o", str(output_dir),
        "--two-stems", "vocals",
        str(audio_path)
    ]
    
    env = os.environ.copy()
    if models_path and os.path.isdir(models_path):
        print(f"使用自定义 Demucs 模型路径: {models_path}")
        env['DEMUCS_MODELS'] = models_path
    
    try:
        print(f"正在执行命令: {' '.join(command)}")
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            print("\nDemucs 运行时发生错误。请查看上面由 Demucs 直接输出的日志信息。")
            return None
        
        print("音源分离完成。")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
        
        if os.path.exists(vocals_path):
            print(f"找到人声音轨: {vocals_path}")
            return vocals_path
        else:
            print(f"错误: 未能找到分离后的人声音轨文件于: {vocals_path}")
            return None
    except Exception as e:
        print(f"运行 Demucs 时发生未知错误: {e}")
        return None

# ==============================================================================
# 3. 核心算法模块
# ==============================================================================

def transcribe_audio(
    audio_path: str,
    model_path_or_size: str,
    device: str,
    compute_type: str
) -> Optional[List[Word]]:
    """
    加载 Whisper 模型并对指定的音频文件进行转写。

    :param audio_path: 用于转写的音频文件路径。
    :param model_path_or_size: Whisper 模型的大小或本地路径。
    :param device: 计算设备 ('cpu', 'cuda')。
    :param compute_type: 计算类型 ('float16', 'int8')。
    :return: 包含所有识别出的词语对象的列表，失败则返回 None。
    """
    print(f"\n--- 语音识别模块 ---")
    print(f"正在加载 Whisper 模型 '{model_path_or_size}'...")
    try:
        model = WhisperModel(model_path_or_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    print(f"正在转写音频 '{os.path.basename(audio_path)}'...")
    start_time = time.time()
    try:
        segments, _ = model.transcribe(audio_path, language="ja", word_timestamps=True)
        all_words = [word for segment in segments for word in segment.words]
    except Exception as e:
        print(f"音频转写过程中发生错误: {e}")
        return None
        
    print(f"转写完成，耗时 {time.time() - start_time:.2f} 秒。共识别出 {len(all_words)} 个词。")
    return all_words

def find_best_match_in_words(
    all_words: List[Word],
    target_sentence: str,
    search_mode: str,
    confidence_threshold: int
) -> Optional[Dict[str, Any]]:
    """
    在已转写的词语列表中，查找与目标句子最匹配的连续词序列。

    :param all_words: Whisper 转写出的词语对象列表。
    :param target_sentence: 要查找的目标句子。
    :param search_mode: 搜索模式 ('efficient' 或 'exhaustive')。
    :param confidence_threshold: 匹配的置信度阈值 (0-100)。
    :return: 包含最佳匹配信息的字典 (时间戳、索引等)，如果未找到则返回 None。
    """
    print(f"\n--- 文本匹配模块 ---")
    norm_target = normalize_japanese_text(target_sentence)
    if not norm_target:
        print("错误：目标句子标准化后为空，无法匹配。")
        return None
    
    norm_words = [normalize_japanese_text(word.word) for word in all_words]
    best_score, best_start_idx, best_end_idx = -1, -1, -1

    if search_mode == 'exhaustive':
        print("使用模式: 穷举搜索 (exhaustive)")
        for i in range(len(norm_words)):
            for j in range(i, len(norm_words)):
                current_sequence = "".join(norm_words[i : j + 1])
                score = fuzz.ratio(norm_target, current_sequence)
                if score > best_score:
                    best_score, best_start_idx, best_end_idx = score, i, j
    else: # efficient mode
        print("使用模式: 高效搜索 (efficient)")
        kks = pykakasi.kakasi()
        try:
            target_word_count = len(kks.convert(target_sentence))
        except:
            target_word_count = len(norm_target) // 3
        min_len, max_len = max(2, int(target_word_count * 0.7)), int(target_word_count * 1.5)
        for i in range(len(norm_words) - min_len + 1):
            for length in range(min_len, max_len + 1):
                j = i + length
                if j > len(norm_words): break
                current_sequence = "".join(norm_words[i:j])
                if not (len(norm_target) * 0.5 < len(current_sequence) < len(norm_target) * 2.0): continue
                score = fuzz.ratio(norm_target, current_sequence)
                if score > best_score:
                    best_score, best_start_idx, best_end_idx = score, i, j - 1
    
    print(f"匹配完成。最高相似度得分: {best_score}")
    if best_score < confidence_threshold:
        print(f"未能找到足够可信的匹配项。最高分 {best_score} 低于阈值 {confidence_threshold}。")
        return None
        
    return {
        "start_timestamp": all_words[best_start_idx].start,
        "end_timestamp": all_words[best_end_idx].end,
        "start_idx": best_start_idx,
        "end_idx": best_end_idx,
        "score": best_score
    }

# ==============================================================================
# 4. 主流程编排模块
# ==============================================================================

def find_and_clip_sentence(
    audio_path: str,
    target_sentence: str,
    output_path: str,
    model_path_or_size: str,
    device: str,
    compute_type: str,
    confidence_threshold: int,
    search_mode: str,
    enable_source_separation: bool,
    demucs_models_path: Optional[str],
    export_transcription_path: Optional[str],
    clip_vocals_only: bool
) -> bool:
    """
    总流程函数，编排音源分离、语音识别、文本匹配和音频裁剪的完整过程。

    (参数注解同上)
    :return: 任务是否成功完成。
    """
    if not os.path.exists(audio_path):
        print(f"错误：音频文件未找到 -> {audio_path}")
        return False

    audio_for_transcription = audio_path
    if enable_source_separation:
        vocals_path = separate_vocals(audio_path, models_path=demucs_models_path)
        if vocals_path:
            audio_for_transcription = vocals_path
        else:
            print("警告: 音源分离失败，将继续使用原始音频进行识别。")
    
    all_words = transcribe_audio(audio_for_transcription, model_path_or_size, device, compute_type)
    if not all_words:
        print("任务终止：语音识别步骤未能返回有效的词语列表。")
        return False

    if export_transcription_path:
        print(f"正在导出完整转写稿到: {export_transcription_path}")
        try:
            with open(export_transcription_path, 'w', encoding='utf-8') as f:
                for word in all_words:
                    f.write(f"[{word.start:.3f} -> {word.end:.3f}] {word.word}\n")
            print("转写稿导出成功。")
        except Exception as e:
            print(f"错误: 导出转写稿失败: {e}")

    match_result = find_best_match_in_words(all_words, target_sentence, search_mode, confidence_threshold)
    if not match_result:
        print("任务终止：文本匹配步骤未能找到匹配项。")
        return False

    print("\n--- 匹配成功! ---")
    print(f"  精确时间范围: 从 {match_result['start_timestamp']:.3f} 秒 到 {match_result['end_timestamp']:.3f} 秒")

    print(f"\n--- 音频裁剪模块 ---")
    final_clip_source_path = audio_path
    if clip_vocals_only and enable_source_separation and audio_for_transcription != audio_path:
        final_clip_source_path = audio_for_transcription
        print(f"裁剪目标: 仅人声 (来自: {os.path.basename(final_clip_source_path)})")
    else:
        print(f"裁剪目标: 原始音频 (来自: {os.path.basename(final_clip_source_path)})")

    try:
        audio = AudioSegment.from_file(final_clip_source_path)
        clipped_audio = audio[int(match_result['start_timestamp'] * 1000):int(match_result['end_timestamp'] * 1000)]
        clipped_audio.export(output_path, format="wav")
        print("--- 任务完成 ---")
        return True
    except Exception as e:
        print(f"裁剪或保存音频时发生错误: {e}")
        return False

# ==============================================================================
# 5. 主程序入口
# ==============================================================================

if __name__ == "__main__":
    # --- 请在这里配置你的任务 ---
    
    INPUT_AUDIO_PATH = r"D:\program\Python\auto-workflows\stt\stt-test\pjsk-test.mp4"
    TARGET_SENTENCE = "今日は、放課後みんなで練習する日だから、 スコア忘れないようにしないと"
    OUTPUT_AUDIO_PATH = "clipped_sentence.wav"

    # --- 音源分离设置 ---
    ENABLE_SOURCE_SEPARATION = True
    DEMUCS_MODELS_PATH = None

    # --- 裁剪设置 ---
    CLIP_VOCALS_ONLY = False

    # --- 调试设置 ---
    EXPORT_TRANSCRIPTION_FILE = True
    TRANSCRIPTION_OUTPUT_PATH = "full_transcription.txt"

    # --- 高级设置 ---
    MODEL_PATH_OR_SIZE = "D:/ACGN/gal/whisper/models/faster-whisper-medium"
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    SEARCH_MODE = 'exhaustive'
    CONFIDENCE_THRESHOLD = 80

    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"错误: 输入文件 '{INPUT_AUDIO_PATH}' 不存在。请检查路径。")
    else:
        find_and_clip_sentence(
            audio_path=INPUT_AUDIO_PATH,
            target_sentence=TARGET_SENTENCE,
            output_path=OUTPUT_AUDIO_PATH,
            model_path_or_size=MODEL_PATH_OR_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            search_mode=SEARCH_MODE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            enable_source_separation=ENABLE_SOURCE_SEPARATION,
            demucs_models_path=DEMUCS_MODELS_PATH,
            export_transcription_path=TRANSCRIPTION_OUTPUT_PATH if EXPORT_TRANSCRIPTION_FILE else None,
            clip_vocals_only=CLIP_VOCALS_ONLY
        )
