# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import sys
import shutil
from typing import List, Dict, Optional, Any

# --- 新增核心依赖 ---
import multiprocessing

# --- 核心依赖 ---
# pip install faster-whisper pydub "thefuzz[speedup]" pykakasi mojimoji demucs

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word
from pydub import AudioSegment
from thefuzz import fuzz
import pykakasi
import mojimoji

# ==============================================================================
# 1. 文本标准化模块 (无变动)
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
# 2. 音源分离模块 (已使用 multiprocessing 重构)
# ==============================================================================

def _separate_vocals_worker(
    queue: multiprocessing.Queue,
    audio_path: str,
    output_dir: str,
    models_path: Optional[str]
):
    """
    【子进程工作函数】使用 Demucs 进行实际的音源分离。
    此函数在独立的子进程中运行，负责加载模型和执行计算密集型任务。
    执行结果通过队列返回给主进程。

    :param queue: 用于与主进程通信的队列。
    :param audio_path: 输入的音频文件路径。
    :param output_dir: 分离后文件的输出目录。
    :param models_path: Demucs 模型文件的本地路径。
    """
    print("子进程: 开始分离人声和背景音乐，此过程可能非常耗时...")
    
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
        print(f"子进程: 使用自定义 Demucs 模型路径: {models_path}")
        env['DEMUCS_MODELS'] = models_path
    
    try:
        print(f"子进程: 正在执行命令: {' '.join(command)}")
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True, encoding='utf-8')
        
        print("子进程: 音源分离完成。")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
        
        if os.path.exists(vocals_path):
            print(f"子进程: 找到人声音轨: {vocals_path}")
            queue.put(vocals_path)
        else:
            print(f"子进程错误: 未能找到分离后的人声音轨文件于: {vocals_path}")
            queue.put(None)
    except subprocess.CalledProcessError as e:
        print("\n子进程错误: Demucs 运行时发生错误。")
        print(f"返回码: {e.returncode}")
        print("--- Demucs STDOUT ---")
        print(e.stdout)
        print("--- Demucs STDERR ---")
        print(e.stderr)
        queue.put(None)
    except Exception as e:
        print(f"子进程错误: 运行 Demucs 时发生未知错误: {e}")
        queue.put(None)

def separate_vocals(audio_path: str, output_dir: str = "temp_separated", models_path: Optional[str] = None) -> Optional[str]:
    """
    【主进程接口】使用 Demucs 将音频文件中的人声和背景音乐分离。
    此函数通过启动一个独立的子进程来执行实际的分离操作，从而将模型加载和
    资源密集型计算与主进程隔离，并在子进程结束后自动回收所有资源（内存/显存）。

    :param audio_path: 输入的音频文件路径。
    :param output_dir: 分离后文件的输出目录。
    :param models_path: Demucs 模型文件的本地路径。
    :return: 分离出的人声音轨文件路径，如果失败则返回 None。
    """
    print("\n--- 音源分离模块 (引擎: Demucs) ---")
    print("正在启动子进程以执行音源分离...")
    
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    process = ctx.Process(target=_separate_vocals_worker, args=(q, audio_path, output_dir, models_path))
    
    process.start()
    result = q.get()
    process.join()

    if result:
        print("主进程: 音源分离子进程成功完成。")
    else:
        print("主进程: 音源分离子进程执行失败。")
        
    return result

# ==============================================================================
# 3. 核心算法模块 (语音识别部分已使用 multiprocessing 重构)
# ==============================================================================

def _transcribe_audio_worker(
    queue: multiprocessing.Queue,
    audio_path: str,
    model_path_or_size: str,
    device: str,
    compute_type: str
):
    """
    【子进程工作函数】加载 Whisper 模型并对音频进行转写。
    此函数在独立的子进程中运行，以隔离模型资源。
    转写结果（词语列表）通过队列返回给主进程。

    :param queue: 用于与主进程通信的队列。
    :param audio_path: 用于转写的音频文件路径。
    :param model_path_or_size: Whisper 模型的大小或本地路径。
    :param device: 计算设备 ('cpu', 'cuda')。
    :param compute_type: 计算类型 ('float16', 'int8')。
    """
    print(f"子进程: 正在加载 Whisper 模型 '{model_path_or_size}' 到设备 '{device}'...")
    try:
        # 在子进程内部导入和加载模型
        from faster_whisper import WhisperModel
        model = WhisperModel(model_path_or_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"子进程错误: 加载模型失败: {e}")
        queue.put(None)
        return

    print(f"子进程: 开始转写音频 '{os.path.basename(audio_path)}'...")
    start_time = time.time()
    try:
        segments, _ = model.transcribe(audio_path, language="ja", word_timestamps=True)
        all_words = [word for segment in segments for word in segment.words]
    except Exception as e:
        print(f"子进程错误: 音频转写过程中发生错误: {e}")
        queue.put(None)
        return
        
    duration = time.time() - start_time
    print(f"子进程: 转写完成，耗时 {duration:.2f} 秒。共识别出 {len(all_words)} 个词。")
    queue.put(all_words)

def transcribe_audio(
    audio_path: str,
    model_path_or_size: str,
    device: str,
    compute_type: str
) -> Optional[List[Word]]:
    """
    【主进程接口】加载 Whisper 模型并对指定的音频文件进行转写。
    此函数通过启动一个独立的子进程来执行实际的转写操作，从而将模型加载和
    资源密集型计算与主进程隔离，并在子进程结束后自动回收所有资源（内存/显存）。

    :param audio_path: 用于转写的音频文件路径。
    :param model_path_or_size: Whisper 模型的大小或本地路径。
    :param device: 计算设备 ('cpu', 'cuda')。
    :param compute_type: 计算类型 ('float16', 'int8')。
    :return: 包含所有识别出的词语对象的列表，失败则返回 None。
    """
    print(f"\n--- 语音识别模块 ---")
    print("正在启动子进程以执行语音识别...")
    
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    process = ctx.Process(target=_transcribe_audio_worker, args=(q, audio_path, model_path_or_size, device, compute_type))
    
    process.start()
    result = q.get()
    process.join()

    if result is not None:
        print(f"主进程: 语音识别子进程成功完成，共返回 {len(result)} 个词。")
    else:
        print("主进程: 语音识别子进程执行失败或未返回结果。")
        
    return result

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
# 4. 最终裁剪函数 (无变动)
# ==============================================================================

def finalize_clip(source_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """
    根据给定的时间戳，从源音频文件中裁剪出片段并保存。

    :param source_path: 用于裁剪的源音频文件路径 (可以是原始文件或纯人声文件)。
    :param start_time: 裁剪的开始时间 (秒)。
    :param end_time: 裁剪的结束时间 (秒)。
    :param output_path: 裁剪后音频的保存路径。
    :return: 裁剪是否成功。
    """
    print(f"\n--- 音频裁剪模块 ---")
    print(f"正在从 '{os.path.basename(source_path)}' 中裁剪...")
    print(f"时间范围: {start_time:.3f}s -> {end_time:.3f}s")
    try:
        audio = AudioSegment.from_file(source_path)
        clipped_audio = audio[int(start_time * 1000):int(end_time * 1000)]
        clipped_audio.export(output_path, format="wav")
        print(f"成功裁剪并保存文件到: {output_path}")
        return True
    except Exception as e:
        print(f"裁剪或保存音频时发生错误: {e}")
        return False

# ==============================================================================
# 5. 主流程编排模块 (接口无变动)
# ==============================================================================

def find_sentence_timestamps(
    audio_path: str,
    target_sentence: str,
    model_path_or_size: str,
    device: str,
    compute_type: str,
    confidence_threshold: int,
    search_mode: str,
    enable_source_separation: bool,
    demucs_models_path: Optional[str],
    export_transcription_path: Optional[str],
    clip_vocals_only: bool
) -> Optional[Dict[str, Any]]:
    """
    执行完整的查找流程，但不进行裁剪，而是返回包含所有结果的字典。

    :param audio_path: 必需，源音频或视频文件的路径。
    :param target_sentence: 必需，希望在音频中定位的日文句子。
    :param model_path_or_size: 必需，faster-whisper 模型的大小 (如 'medium') 或本地路径。
    :param device: 必需，计算设备，如 'cuda' 或 'cpu'。
    :param compute_type: 必需，计算类型，如 'float16' (推荐 GPU) 或 'int8'。
    :param confidence_threshold: 必需，匹配置信度阈值 (0-100)，低于此分数的匹配将被忽略。
    :param search_mode: 必需，搜索模式，'efficient' (高效) 或 'exhaustive' (穷举)。
    :param enable_source_separation: 必需，布尔值，是否启用 Demucs 进行人声分离。
    :param demucs_models_path: 可选，Demucs 模型的自定义本地路径。
    :param export_transcription_path: 可选，若提供路径，则会将完整的转写稿保存到该文件。
    :param clip_vocals_only: 必需，布尔值，如果启用了人声分离，最终是否从分离出的人声音轨中裁剪。
    :return: 包含查找结果的字典 {'start_time': float, 'end_time': float, 'clip_source_path': str}，如果失败则返回 None。
    """
    if not os.path.exists(audio_path):
        print(f"错误：音频文件未找到 -> {audio_path}")
        return None

    audio_for_transcription = audio_path
    vocals_path_if_separated = None
    if enable_source_separation:
        vocals_path = separate_vocals(audio_path, models_path=demucs_models_path)
        if vocals_path:
            audio_for_transcription = vocals_path
            vocals_path_if_separated = vocals_path
        else:
            print("警告: 音源分离失败，将继续使用原始音频进行识别。")
    
    all_words = transcribe_audio(audio_for_transcription, model_path_or_size, device, compute_type)
    if not all_words:
        print("任务终止：语音识别步骤未能返回有效的词语列表。")
        return None

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
        return None

    final_clip_source_path = audio_path
    if clip_vocals_only and vocals_path_if_separated:
        final_clip_source_path = vocals_path_if_separated
        
    return {
        "start_time": match_result['start_timestamp'],
        "end_time": match_result['end_timestamp'],
        "clip_source_path": final_clip_source_path
    }

# ==============================================================================
# 6. 主程序入口 (无变动)
# ==============================================================================

if __name__ == "__main__":
    # 确保在 Windows 或 macOS 上，multiprocessing 的 'spawn' 或 'forkserver' 模式正常工作
    multiprocessing.freeze_support()
    
    # --- 配置任务 (直接运行时使用) ---
    
    INPUT_AUDIO_PATH = r"D:\program\Python\auto-workflows\stt\stt-test\pjsk-test.mp4"
    TARGET_SENTENCE = "今日は、放課後みんなで練習する日だから、 スコア忘れないようにしないと"
    OUTPUT_AUDIO_PATH = "clipped_sentence.wav"

    ENABLE_SOURCE_SEPARATION = True
    DEMUCS_MODELS_PATH = None
    CLIP_VOCALS_ONLY = False
    EXPORT_TRANSCRIPTION_FILE = True
    TRANSCRIPTION_OUTPUT_PATH = "full_transcription.txt"

    MODEL_PATH_OR_SIZE = "D:/ACGN/gal/whisper/models/faster-whisper-medium"
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    SEARCH_MODE = 'exhaustive'
    CONFIDENCE_THRESHOLD = 80

    print("--- 开始执行直接裁剪任务 (已启用多进程模式) ---")
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"错误: 输入文件 '{INPUT_AUDIO_PATH}' 不存在。请检查路径。")
    else:
        # 1. 调用主流程函数获取 AI 预测的时间戳和裁剪源
        result = find_sentence_timestamps(
            audio_path=INPUT_AUDIO_PATH,
            target_sentence=TARGET_SENTENCE,
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
        
        # 2. 如果成功找到，立即执行裁剪
        if result:
            print("\n--- AI 预测成功，立即执行裁剪 ---")
            finalize_clip(
                source_path=result["clip_source_path"],
                start_time=result["start_time"],
                end_time=result["end_time"],
                output_path=OUTPUT_AUDIO_PATH
            )
        else:
            print("\n--- 任务结束，未能找到匹配项或发生错误，未执行裁剪。 ---")
