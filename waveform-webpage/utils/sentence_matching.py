# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess
import sys
import shutil
from typing import List, Dict, Optional, Any, Tuple

# --- 核心依赖 ---
import multiprocessing
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word
from thefuzz import fuzz
import pykakasi
import mojimoji

try:
    # 尝试相对导入，这在作为模块导入时会成功
    from .fast_match import fast_fuzzy_match
    from .llm_handler import llm_normalize
except ImportError:
    # 如果相对导入失败，说明是作为顶层脚本运行，回退到绝对导入
    from fast_match import fast_fuzzy_match
    from llm_handler import llm_normalize


SUPPRESS_TOKEN_FILE = r"D:\program\Python\auto-workflows\stt-backup\stt-test\prevent-kanji-stt\kanji_tokens.txt"
PROMPT = "いかの たいわは ぜんぶ ひらがなか カタカナで アウトプットして ください。かんじは ぜったい ダメです。では、はじめましょう。"

# --- 【核心修正】将文件名清理函数移入此文件 ---
def _sanitize_filename_part(text: str, max_length: int = 50) -> str:
    """
    清理字符串，使其成为一个安全的文件名组成部分。
    移除所有在主流操作系统中不允许用于文件名的字符，并截断到合理长度。
    """
    # 定义一个包含常见非法字符的正则表达式 (包括全角和半角版本)
    invalid_chars = r'[\\/:*?"<>|？：]'
    # 将所有非法字符替换为下划线
    sanitized_text = re.sub(invalid_chars, '_', text)
    # 将可能存在的空白符（如换行、制表符）也替换为下划线
    sanitized_text = re.sub(r'\s+', '_', sanitized_text)
    # 截断文件名，避免过长导致的问题
    return sanitized_text[:max_length]

# ==============================================================================
# 1. 文本标准化模块 (有修改)
# ==============================================================================
def normalize_japanese_text(text: str) -> str:
    """标准化日文文本为纯平假名。"""
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    hiragana_text = "".join([item['hira'] for item in result])
    normalized_text = mojimoji.zen_to_han(hiragana_text, kana=False)
    cleaned_text = re.sub(r'[^ぁ-ん]', '', normalized_text)
    return cleaned_text

def normalize_text_llm(text: str) -> str:
    """使用 LLM 标准化单个日文文本，失败则回退。"""
    try:
        # llm_normalize 需要列表并返回列表
        normalized_list = llm_normalize([text])
        return normalized_list[0]
    except Exception as e:
        print(f"LLM 标准化失败，回退到传统方法: {e}")
        return normalize_japanese_text(text)

def batch_normalize_texts(texts: List[str]) -> Dict[str, str]:
    """
    【新增】一次性批量标准化所有文本，并返回原文到标准化的映射字典。
    """
    print(f"准备使用 LLM 批量标准化 {len(texts)} 条文本...")
    try:
        normalized_texts = llm_normalize(texts)
        if len(texts) != len(normalized_texts):
             raise Exception("LLM返回的列表长度与输入不匹配")
        return {original: normalized for original, normalized in zip(texts, normalized_texts)}
    except Exception as e:
        print(f"LLM 批量标准化失败，将对每个句子回退到传统方法: {e}")
        return {text: normalize_japanese_text(text) for text in texts}


# ==============================================================================
# 2. 音源分离与语音识别模块 (有修改)
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
        # --- 【核心修改】新增参数以控制是否执行标准化 ---
        perform_normalization: bool,
        suppress_file_path: Optional[str]=SUPPRESS_TOKEN_FILE,
        prompt: str=PROMPT
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
            initial_prompt=prompt,
            condition_on_previous_text=False,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=500),
            **kwargs
        )
        all_words: List[Word] = []
        print("子进程: 实时转写进度...")
        for segment in segments:
            start_time_str = time.strftime('%M:%S', time.gmtime(segment.start))
            print(f"  [{start_time_str}] {segment.text.strip()}")
            if segment.words:
                all_words.extend(segment.words)
        
        if not all_words:
            raise Exception("错误：未识别出任何词语")

        # --- 【核心修改】根据参数决定是否执行标准化 ---
        if perform_normalization:
            print("子进程: 转写完成，开始标准化转写稿...")
            try:
                word_texts = [word.word for word in all_words]
                normalized_word_texts = llm_normalize(word_texts)
            except Exception as e:
                print(f"子进程: LLM 标准化失败，回退到传统方法: {e}")
                normalized_word_texts = [normalize_japanese_text(word.word) for word in all_words]

            print("子进程: 标准化完成。")
            queue.put((all_words, normalized_word_texts))
        else:
            print("子进程: 转写完成。")
            queue.put((all_words, None)) # 保持元组结构，第二个元素为 None

    except Exception as e:
        print(f"子进程 Whisper 错误: {e}")
        queue.put(None)

def transcribe_and_normalize_audio(
    audio_path: str, 
    model_path: str, 
    device: str, 
    compute_type: str,
    # --- 【核心修改】新增参数，并设默认值为 True 以保持向后兼容 ---
    perform_normalization: bool = True
) -> Optional[Tuple[List[Word], Optional[List[str]]]]:
    """
    【修改】主进程接口，启动子进程进行语音识别和可选的标准化。
    返回原始词对象和（可选的）标准化后的词文本列表。
    """
    print("\n--- 语音识别模块 ---")
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()
    process = ctx.Process(
        target=_transcribe_audio_worker,
        # --- 【核心修改】将新参数传递给子进程 ---
        args=(q, audio_path, model_path, device, compute_type, perform_normalization, SUPPRESS_TOKEN_FILE)
    )
    process.start()
    result = q.get()
    process.join()
    print("主进程: 语音识别完成。")
    return result

def normalize_words(all_words: list[Word], method: str = "llm") -> list[str]:
    """
    句子标准化函数接口。
    如果指定使用 "llm" 方法，将通过网络调用 LLM 并获取输出
    """
    if method == "normal":
        print("使用传统方法进行标准化...")
        return [normalize_japanese_text(word.word) for word in all_words]
    elif method == "llm":
        print("使用 LLM 进行标准化...")
        try:
            return llm_normalize([word.word for word in all_words])
        except Exception as e:
            print(f"LLM 批量标准化失败，回退到传统逐词标准化方法: {e}")
            return [normalize_japanese_text(word.word) for word in all_words]
    else:
        raise NotImplementedError()

# ==============================================================================
# 3. 核心匹配算法 (有修改)
# ==============================================================================
def find_best_match_in_words(
    norm_words: List[str],
    norm_target: str,
    all_words: List[Word],
    search_mode: str,
    confidence_threshold: int
) -> Optional[Dict[str, Any]]:
    """
    【修改】在已转写的词语列表中，为单个句子查找最匹配的序列。
    此函数不再执行标准化，而是接收预标准化的文本。
    """
    if not norm_target:
        return None
    
    best_score, best_start_idx, best_end_idx = -1, -1, -1

    if search_mode == 'exhaustive':
        for i in range(len(norm_words)):
            for j in range(i, len(norm_words)):
                current_sequence = "".join(norm_words[i : j + 1])
                score = fuzz.ratio(norm_target, current_sequence)
                if score > best_score:
                    best_score, best_start_idx, best_end_idx = score, i, j
    elif search_mode == 'efficient':
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
    elif search_mode == "levenschtein":
        best_score, best_start_idx, best_end_idx = fast_fuzzy_match(norm_words, norm_target)
    
    if best_score < confidence_threshold:
        return None
        
    return {
        "start_timestamp": all_words[best_start_idx].start - 0.3,
        "end_timestamp": all_words[best_end_idx].end,
        "matched_text": "".join([w.word for w in all_words[best_start_idx:best_end_idx+1]]),
        "score": best_score
    }

# ==============================================================================
# 4. 最终裁剪函数 (FFmpeg 加速)
# ==============================================================================
def finalize_clip(
    source_path: str, start_time: float, end_time: float, output_path: str,
    balance_volume: bool = False,
    volume_i: float = -19.1, volume_lra: float = 11, volume_tp = -1.5,
) -> bool:
    """
    使用 FFmpeg 直接裁剪音频
    同时完成音量平衡操作
    """
    print(f"\n--- FFmpeg 快速裁剪模块 ---")
    print(f"正在从 '{os.path.basename(source_path)}' 裁剪: {start_time:.3f}s -> {end_time:.3f}s")

    try:
        # 计算需要裁剪的持续时间
        duration = end_time - start_time
        if duration <= 0:
            print("错误：结束时间必须大于开始时间。")
            return False

        if balance_volume:
            clip_output_path = os.path.splitext(output_path)[0] + "_clipped.wav"
        else:
            clip_output_path = output_path

        # 构建 FFmpeg 命令
        # -ss [start_time]: 定位到开始时间 (放在 -i 前面可以实现快速寻址)
        # -i [source_path]: 输入文件
        # -t [duration]: 从开始时间起，处理多长的音频
        # -vn: 忽略视频流
        # -acodec pcm_s16le: 输出为标准的 WAV 格式 (16-bit PCM)
        # -y: 如果输出文件已存在，则自动覆盖
        # -hide_banner -loglevel error: 精简输出，只显示错误信息
        command = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error',
            '-ss', str(start_time),
            '-i', source_path,
            '-t', str(duration),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-y',
            clip_output_path
        ]

        # 执行命令
        subprocess.run(command, check=True, capture_output=True, text=True)

        if balance_volume:
            balance_volume_cmd = [
                'ffmpeg',
                "-y",               # Overwrite temp file if exists
                "-i", clip_output_path,
                "-filter:a", f"loudnorm=I={volume_i}:LRA={volume_lra}:TP={volume_tp}",
                "-vn",              # No video
                output_path
            ]
            subprocess.run(
                balance_volume_cmd, check=True, capture_output=True, text=True,encoding='utf-8'
            )

        print(f"成功保存到: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        # 如果 FFmpeg 返回非零退出码，说明出错了
        print(f"FFmpeg 裁剪或平衡音量错误: {e.stderr}")
        return False
    except Exception as e:
        # 捕获其他可能的错误，如文件未找到
        print(f"执行裁剪时发生未知错误: {e}")
        return False

# ==============================================================================
# 5. 主流程编排模块 (有修改)
# ==============================================================================
def find_best_match_for_all_sentences(
    all_words: List[Word],
    norm_words: List[str],
    target_sentences_map: Dict[str, str],
    confidence_threshold: int,
    search_mode: str
) -> List[Dict[str, Any]]:
    """
    【修改】接收预标准化的转写稿和句子映射，返回所有成功匹配的结果。
    """
    all_results = []
    if not all_words or not norm_words:
        return []
        
    print(f"开始在标准化的转写稿中匹配 {len(target_sentences_map)} 个句子...")
    for sentence, norm_sentence in target_sentences_map.items():
        match_result = find_best_match_in_words(
            norm_words=norm_words,
            norm_target=norm_sentence,
            all_words=all_words,
            search_mode=search_mode,
            confidence_threshold=confidence_threshold
        )
        if match_result:
            print(f"  -> 成功匹配: '{sentence[:20]}...' (得分: {match_result['score']})")
            safe_id_part = _sanitize_filename_part(sentence[:20], max_length=20)
            all_results.append({
                "id": f"clip-{safe_id_part}-{time.time():.8f}",
                "sentence": sentence,
                "predicted_start": round(match_result['start_timestamp'], 3),
                "predicted_end": round(match_result['end_timestamp'], 3),
                "score": match_result['score']
            })
    return all_results

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
    【修改】针对单个音频文件和多个句子，执行优化的完整流程。
    """
    if not os.path.exists(audio_path):
        print(f"错误：文件未找到 -> {audio_path}")
        return None

    def update_progress(message: str):
        print(message)
        if progress_callback:
            progress_callback(message)

    audio_for_transcription = audio_path
    vocals_path_if_separated = None
    if enable_source_separation:
        update_progress("步骤 1/4: 正在进行音源分离...")
        vocals_path = separate_vocals(audio_path, models_path=demucs_models_path)
        if vocals_path:
            audio_for_transcription = vocals_path
            vocals_path_if_separated = vocals_path
        else:
            update_progress("警告: 音源分离失败，使用原始音频。")

    update_progress("步骤 2/4: 正在进行语音识别与转写稿标准化...")
    transcription_result = transcribe_and_normalize_audio(audio_for_transcription, model_path_or_size, device, compute_type)
    if not transcription_result:
        update_progress("错误: 语音识别步骤失败。任务终止。")
        return {"all_words": None, "norm_words": None, "clip_source_path": None}
    
    all_words, norm_words = transcription_result
    
    update_progress("步骤 3/4: 正在批量标准化目标句子...")
    target_sentences_map = batch_normalize_texts(target_sentences)
    if not target_sentences_map:
        update_progress("错误: 目标句子标准化失败。任务终止。")
        return {"all_words": all_words, "norm_words": norm_words, "clip_source_path": None}

    final_clip_source_path = audio_path
    if clip_vocals_only and vocals_path_if_separated:
        final_clip_source_path = vocals_path_if_separated

    # 仅转写模式 (供 app.py 调用)
    if not target_sentences:
        return {
            "all_words": all_words,
            "norm_words": norm_words,
            "clip_source_path": final_clip_source_path
        }
    
    # 完整流程模式 (供直接测试)
    update_progress("步骤 4/4: 正在匹配所有句子...")
    all_results = find_best_match_for_all_sentences(
        all_words=all_words,
        norm_words=norm_words,
        target_sentences_map=target_sentences_map,
        confidence_threshold=confidence_threshold,
        search_mode=search_mode
    )
    update_progress(f"处理完成，成功匹配 {len(all_results)}/{len(target_sentences)} 个句子。")

    return {
        "clip_source_path": final_clip_source_path,
        "clips": all_results
    }


# ==============================================================================
# 6. 主程序入口 (无变动)
# ==============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    INPUT_AUDIO_PATH = "D:/program/Python/auto-workflows/stt/stt-test/pjsk-test.mp4"
    TARGET_SENTENCES = [
        "スコア忘れないようにしないと",
        "今日は、放課後みんなで練習する日だから",
        "この前のライブ、すっごく盛り上がったよね"
    ]
    OUTPUT_DIR = "test_output_clips"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ENABLE_SOURCE_SEPARATION = True
    DEMUCS_MODELS_PATH = None
    CLIP_VOCALS_ONLY = False

    MODEL_PATH_OR_SIZE = "D:/ACGN/gal/whisper/models/faster-whisper-medium"
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    # SEARCH_MODE = 'exhaustive'
    SEARCH_MODE = 'levenschtein'
    CONFIDENCE_THRESHOLD = 50

    print("--- 开始执行批量处理测试 (串行模式) ---")
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"错误: 输入文件 '{INPUT_AUDIO_PATH}' 不存在。请检查路径。")
    else:
        results_data = find_multiple_sentences_timestamps(
            audio_path=INPUT_AUDIO_PATH,
            target_sentences=TARGET_SENTENCES,
            model_path_or_size=MODEL_PATH_OR_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            search_mode=SEARCH_MODE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            enable_source_separation=ENABLE_SOURCE_SEPARATION,
            demucs_models_path=DEMUCS_MODELS_PATH,
            clip_vocals_only=CLIP_VOCALS_ONLY
        )
        
        if results_data and results_data.get("clips"):
            print(f"\n--- AI 预测成功，共找到 {len(results_data['clips'])} 个匹配项，开始逐一裁剪 ---")
            for i, clip_info in enumerate(results_data['clips']):
                print(f"\n--- 正在裁剪第 {i+1} 个片段 ---")
                print(f"句子: {clip_info['sentence']}")
                output_filename = f"clip_{i}_{re.sub('[^ぁ-んァ-ン一-龥]', '', clip_info['sentence'][:10])}.wav"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                finalize_clip(
                    source_path=results_data["clip_source_path"],
                    start_time=clip_info["predicted_start"],
                    end_time=clip_info["predicted_end"],
                    output_path=output_path
                )
        else:
            print("\n--- 任务结束，未能找到匹配项或发生错误，未执行裁剪。 ---")
