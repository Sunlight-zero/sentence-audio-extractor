# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import sys
import shutil
from typing import List, Dict, Optional, Any

# --- 核心依赖 ---
import multiprocessing
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word
from pydub import AudioSegment
from thefuzz import fuzz
import pykakasi
import mojimoji

from .fuzzy_string_matching import fuzzy_match


SUPPRESS_TOKEN_FILE = None

# ==============================================================================
# 1. 文本标准化模块 (无变动)
# ==============================================================================
def normalize_japanese_text(text: str) -> str:
    """标准化日文文本为纯平假名。"""
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    hiragana_text = "".join([item['hira'] for item in result])
    normalized_text = mojimoji.zen_to_han(hiragana_text, kana=False)
    cleaned_text = re.sub(r'[^ぁ-ん]', '', normalized_text)
    return cleaned_text

# ==============================================================================
# 2. 音源分离与语音识别模块 (使用多进程隔离资源)
# ==============================================================================
def _separate_vocals_worker(queue: multiprocessing.Queue, audio_path: str, output_dir: str, models_path: Optional[str]):
    """【子进程】执行Demucs音源分离。"""
    print("子进程: 开始分离人声...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    command = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "-o", str(output_dir), "--two-stems", "vocals", str(audio_path)]
    env = os.environ.copy()
    if models_path:
        env['DEMUCS_MODELS'] = models_path
    try:
        # --- 关键修改: 增加 errors='ignore' 来防止因编码问题导致的崩溃 ---
        subprocess.run(command, env=env, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
        queue.put(vocals_path if os.path.exists(vocals_path) else None)
    except Exception as e:
        print(f"子进程 Demucs 错误: {e}")
        queue.put(None)

def separate_vocals(audio_path: str, output_dir: str = "temp_separated", models_path: Optional[str] = None) -> Optional[str]:
    """【主进程接口】启动子进程进行音源分离。"""
    print("\n--- 音源分离模块 ---")
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    process = ctx.Process(target=_separate_vocals_worker, args=(q, audio_path, output_dir, models_path))
    process.start()
    result = q.get()
    process.join()
    print("主进程: 音源分离完成。")
    return result

def load_suppress_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [int(token) for token in f.read().split()]

def _transcribe_audio_worker(
        queue: multiprocessing.Queue, 
        audio_path: str, model_path: str, 
        device: str, compute_type: str,
        suppress_file_path: Optional[str]=SUPPRESS_TOKEN_FILE
    ):
    """【子进程】执行Whisper语音识别。"""
    print(f"子进程: 加载 Whisper 模型 '{model_path}'...")
    try:
        kwargs = dict()
        if suppress_file_path:
            suppress_tokens = load_suppress_tokens(suppress_file_path)
            kwargs["suppress_tokens"] = suppress_tokens
        
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        print(f"子进程: 开始转写音频 '{os.path.basename(audio_path)}'...")
        segments, _ = model.transcribe(
            audio_path, 
            language="ja", 
            word_timestamps=True,
            **kwargs
        )
        all_words = []
        print("子进程: 实时转写进度...")
        for segment in segments:
            # 格式化时间戳为 MM:SS
            start_time_str = time.strftime('%M:%S', time.gmtime(segment.start))
            # 打印进度
            print(f"  [{start_time_str}] {segment.text.strip()}")
            # 收集词语以便后续处理
            if segment.words:
                all_words.extend(segment.words)
        
        if len(all_words) == 0:
            raise Exception("错误：未识别出任何词语")
        print(f"子进程: 转写完成，识别出 {len(all_words)} 个词。")
        sentence = ''.join(word.word for word in all_words)
        print(f"识别的句子为：{sentence}")
        queue.put(all_words)
    except Exception as e:
        print(f"子进程 Whisper 错误: {e}")
        queue.put(None)

def transcribe_audio(audio_path: str, model_path: str, device: str, compute_type: str) -> Optional[List[Word]]:
    """【主进程接口】启动子进程进行语音识别。"""
    print("\n--- 语音识别模块 ---")
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    process = ctx.Process(
        target=_transcribe_audio_worker,
        args=(q, audio_path, model_path, device, compute_type, SUPPRESS_TOKEN_FILE)
    )
    process.start()
    result = q.get()
    process.join()
    print("主进程: 语音识别完成。")
    return result

# ==============================================================================
# 3. 核心匹配算法 (无变动)
# ==============================================================================
def find_best_match_in_words(
    all_words: List[Word],
    target_sentence: str,
    search_mode: str,
    confidence_threshold: int
) -> Optional[Dict[str, Any]]:
    """在已转写的词语列表中，为单个句子查找最匹配的序列。"""
    print(f"\n--- 正在匹配句子: '{target_sentence[:30]}...' ---")
    norm_target = normalize_japanese_text(target_sentence)
    if not norm_target:
        print("错误：目标句子标准化后为空。")
        return None
    
    norm_words = [normalize_japanese_text(word.word) for word in all_words]
    sentence = ''.join(norm_words)
    # print(f"标准化的原始音频为：{sentence}")
    best_score, best_start_idx, best_end_idx = -1, -1, -1

    if search_mode == 'exhaustive':
        print("使用模式: 穷举搜索 (exhaustive)")
        for i in range(len(norm_words)):
            for j in range(i, len(norm_words)):
                current_sequence = "".join(norm_words[i : j + 1])
                score = fuzz.ratio(norm_target, current_sequence)
                if score > best_score:
                    best_score, best_start_idx, best_end_idx = score, i, j
    elif search_mode == 'efficient':  # 'efficient' mode
        print("使用模式: 高效搜索 (efficient)")
        target_len = len(norm_target)
        min_len = max(1, int(target_len * 0.6))
        max_len = int(target_len * 1.6)
        
        for i in range(len(norm_words)):
            for length in range(min_len, max_len + 1):
                j = i + length
                if j > len(norm_words):
                    break
                current_sequence = "".join(norm_words[i:j])
                score = fuzz.ratio(norm_target, current_sequence)
                if score > best_score:
                    best_score, best_start_idx, best_end_idx = score, i, j - 1
    else: # 'levenshtein' 编辑距离模式
        # TODO
        # best_start_idx, best_end_idx, best_score = fuzzy_match(...)
        pass
    
    print(f"最高相似度得分: {best_score}")
    if best_score < confidence_threshold:
        print(f"-> 匹配失败: 最高分 {best_score} 低于阈值 {confidence_threshold}。")
        return None
        
    print(f"-> 匹配成功!")
    return {
        "start_timestamp": all_words[best_start_idx].start - 0.3, # 手动往前调整
        "end_timestamp": all_words[best_end_idx].end,
        "matched_text": "".join([w.word for w in all_words[best_start_idx:best_end_idx+1]]),
        "score": best_score
    }

# ==============================================================================
# 4. 最终裁剪函数 (无变动)
# ==============================================================================
def finalize_clip(source_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """根据时间戳裁剪音频。"""
    print(f"\n--- 音频裁剪模块 ---")
    print(f"正在从 '{os.path.basename(source_path)}' 裁剪: {start_time:.3f}s -> {end_time:.3f}s")
    try:
        audio = AudioSegment.from_file(source_path)
        clipped_audio = audio[int(start_time * 1000):int(end_time * 1000)]
        clipped_audio.export(output_path, format="wav")
        print(f"成功保存到: {output_path}")
        return True
    except Exception as e:
        print(f"裁剪错误: {e}")
        return False

# ==============================================================================
# 5. 主流程编排模块 (保留原始函数，新增批量处理函数)
# ==============================================================================

# --- 【原始函数，已保留】---
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
    (注释与原始代码一致)
    """
    # ... 函数实现与之前版本一致，此处省略以保持简洁 ...
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
        # ...
        pass
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
        "clip_source_path": final_clip_source_path,
        "score": match_result.get('score', 0)
    }


# --- 【新增函数】智能批量处理主流程 ---
def find_multiple_sentences_timestamps(
    audio_path: str,
    target_sentences: List[str],
    model_path_or_size: str,
    device: str,
    compute_type: str,
    confidence_threshold: int,
    search_mode: str,
    enable_source_separation: bool,
    clip_vocals_only: bool,
    demucs_models_path: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Optional[Dict[str, Any]]:
    """
    针对单个音频文件，智能处理多个目标句子。
    (注释与之前版本一致)
    """
    if not os.path.exists(audio_path):
        print(f"错误：文件未找到 -> {audio_path}")
        return None

    def update_progress(message: str):
        print(message)
        if progress_callback:
            progress_callback(message)

    # ... 函数实现与之前版本一致，此处省略以保持简洁 ...
    audio_for_transcription = audio_path
    vocals_path_if_separated = None
    if enable_source_separation:
        update_progress("步骤 1/3: 正在进行音源分离 (此过程较慢)...")
        vocals_path = separate_vocals(audio_path, models_path=demucs_models_path)
        if vocals_path:
            audio_for_transcription = vocals_path
            vocals_path_if_separated = vocals_path
            update_progress("音源分离成功，将使用人声轨道进行识别。")
        else:
            update_progress("警告: 音源分离失败，将使用原始音频进行识别。")
    update_progress("步骤 2/3: 正在进行语音识别 (此过程较慢)...")
    all_words = transcribe_audio(audio_for_transcription, model_path_or_size, device, compute_type)
    if not all_words:
        update_progress("错误: 语音识别步骤未能返回有效的词语列表。任务终止。")
        return None
    update_progress("步骤 3/3: 正在匹配所有句子...")
    all_results = []
    total_sentences = len(target_sentences)
    for i, sentence in enumerate(target_sentences):
        update_progress(f"正在匹配 ({i+1}/{total_sentences}): {sentence[:25]}...")
        match_result = find_best_match_in_words(
            all_words=all_words,
            target_sentence=sentence,
            search_mode=search_mode,
            confidence_threshold=confidence_threshold
        )
        if match_result:
            # ==========================================================
            # === 核心修改：将时间戳四舍五入到三位小数 ===
            # ==========================================================
            all_results.append({
                "id": f"clip-{i}",
                "sentence": sentence,
                "predicted_start": round(match_result['start_timestamp'], 3),
                "predicted_end": round(match_result['end_timestamp'], 3),
                "score": match_result['score']
            })
            # ==========================================================
    if not all_results:
        update_progress("处理完成，但未能找到任何可信的匹配项。")
    else:
        update_progress(f"处理完成，成功匹配 {len(all_results)}/{total_sentences} 个句子。")
    final_clip_source_path = audio_path
    if clip_vocals_only and vocals_path_if_separated:
        final_clip_source_path = vocals_path_if_separated
        print(f"最终裁剪将使用分离出的人声音轨: {final_clip_source_path}")
    else:
        print(f"最终裁剪将使用原始文件: {final_clip_source_path}")
    return {
        "clip_source_path": final_clip_source_path,
        "clips": all_results
    }

# ==============================================================================
# 6. 主程序入口 (无变动)
# ==============================================================================
if __name__ == "__main__":
    # 确保在 Windows 或 macOS 上，multiprocessing 的 'spawn' 或 'forkserver' 模式正常工作
    multiprocessing.freeze_support()
    
    # --- 配置任务 (直接运行时使用) ---
    
    INPUT_AUDIO_PATH = r"D:\program\Python\auto-workflows\stt\stt-test\pjsk-test.mp4"
    TARGET_SENTENCE = "スコア忘れないようにしないと"
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
    CONFIDENCE_THRESHOLD = 75

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
