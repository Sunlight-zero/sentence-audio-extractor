# app.py
import os
import json
import uuid
import traceback
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from waitress import serve
import threading
# --- 【新增】导入并行处理所需的库 ---
import queue
import concurrent.futures
from typing import List, Dict, Set, Any, Tuple, Optional

# 确保 utils 目录在 Python 路径中
import utils.sentence_matching as sm
# --- 新增 ---
# 导入新的 Anki 处理模块
import utils.anki_handler as anki
# --- 【核心修改】移除不再需要的直接导入 ---


# --- Flask 应用初始化与目录配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'tmp/uploads')
RESULTS_FOLDER = os.path.join(APP_ROOT, 'tmp/results')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'tmp/output')
SENTENCES_FOLDER = os.path.join(APP_ROOT, 'tmp/sentences')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SENTENCES_FOLDER, exist_ok=True) # --- 新增 ---
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SENTENCES_FOLDER'] = SENTENCES_FOLDER

# 使用字典来跟踪后台任务状态
tasks = {}

# ==============================================================================
# 页面服务路由
# ==============================================================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/proofread')
def proofread_page():
    return send_from_directory('static', 'proofread.html')

# ==============================================================================
# API 端点
# ==============================================================================

# --- 新增 API: 从 Anki 获取句子 ---
@app.route('/api/anki/sentences', methods=['GET'])
def get_anki_sentences():
    """从 AnkiConnect 获取需要处理的句子。"""
    try:
        deck_name = request.args.get('deck', 'luna temporary') # 允许前端指定牌组
        sentences_path = os.path.join(app.config['SENTENCES_FOLDER'], 'anki_sentences')
        result = anki.extract_sentences_from_anki(path=sentences_path, deck_name=deck_name)
        return jsonify({
            "success": True,
            "sentences": result["sentences_text"],
            "message": f"成功从牌组 '{deck_name}' 提取 {len(result['id_to_sentence_map'])} 个句子。"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# --- 修改 API: 处理音视频文件 ---
@app.route('/process', methods=['POST'])
def process_videos():
    """接收一个或多个视频和多个句子，启动后台批量分析任务。"""
    video_files = request.files.getlist('videoFiles')
    if not video_files or all(f.filename == '' for f in video_files):
        return jsonify({"error": "缺少视频文件或未选择文件。"}), 400
    
    sentences_text = request.form.get('sentence', '')
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        return jsonify({"error": "目标句子不能为空。"}), 400

    enable_separation = request.form.get('separateVocals') == 'on'

    video_paths = []
    for video_file in video_files:
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        video_paths.append(video_path)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "message": "任务已开始，正在初始化..."}

    thread = threading.Thread(target=run_processing_task, args=(task_id, video_paths, sentences, enable_separation))
    thread.start()

    return jsonify({"status": "processing", "task_id": task_id})

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """获取指定后台任务的状态。"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found", "error": "任务ID不存在。"}), 404
    return jsonify(task)

@app.route('/clip', methods=['POST'])
def clip_audio():
    """接收校对后的时间戳，执行最终裁剪。"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    clip_source_path = data.get('clip_source_path')
    original_video_filename = data.get('original_video_filename')
    sentence_id = data.get('sentence_id', 'clip')

    if None in [start_time, end_time, clip_source_path, original_video_filename]:
        return jsonify({"error": "请求中缺少必要的参数。"}), 400

    if not os.path.exists(clip_source_path):
         return jsonify({"error": f"裁剪源文件未找到: {clip_source_path}"}), 404

    try:
        base_name = os.path.splitext(original_video_filename)[0]
        output_filename = f"{base_name}_{sentence_id}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        success = sm.finalize_clip(
            source_path=clip_source_path,
            start_time=float(start_time),
            end_time=float(end_time),
            output_path=output_path
        )
        
        if success:
            return jsonify({
                "message": "音频裁剪成功！", 
                "download_url": url_for('serve_output', filename=output_filename)
            })
        else:
            return jsonify({"error": "后端裁剪音频失败。"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"裁剪时发生服务器错误: {e}"}), 500

# --- 新增 API: 上传校对好的音频到 Anki ---
@app.route('/api/anki/upload', methods=['POST'])
def upload_to_anki():
    """接收前端校对好的片段数据，并调用 anki_handler 上传。"""
    try:
        clips_to_upload = request.json.get('clips')
        target_note_type: Optional[str] = request.json.get('target_note_type')
        if not clips_to_upload:
            return jsonify({"success": False, "error": "没有提供可上传的片段。"}), 400
        
        anki.upload_clips_to_anki(clips_to_upload, target_note_type=target_note_type)
        
        return jsonify({"success": True, "message": "上传任务已成功提交到 Anki。"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================================================================
# 文件服务路由
# ==============================================================================
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ==============================================================================
# 后台任务函数 (核心修改处)
# ==============================================================================
def run_processing_task(
        task_id: str, 
        video_paths: List[str], 
        sentences: List[str], 
        enable_separation: bool
    ) -> None:
    """
    使用三级生产者-消费者模型实现GPU、LLM、CPU并行处理。
    """
    try:
        # --- 1. 初始化与预处理 ---
        sentence_map_path = os.path.join(app.config['SENTENCES_FOLDER'], 'anki_sentences.json')
        sentence_to_note_id = {}
        if os.path.exists(sentence_map_path):
            with open(sentence_map_path, 'r', encoding='utf-8') as f:
                id_to_sentence: Dict[int, str] = json.load(f)
                sentence_to_note_id = {v: k for k, v in id_to_sentence.items()}
        
        tasks[task_id]["message"] = f"正在异步批量标准化 {len(sentences)} 个目标句子..."
        preprocess_queue = queue.Queue(maxsize=1)
        
        def preprocess_worker():
            """在独立线程中运行目标句子的LLM标准化。"""
            try:
                normalized_map = sm.batch_normalize_texts(sentences)
                preprocess_queue.put(normalized_map)
            except Exception as e:
                print(f"LLM 句子标准化工作线程发生错误: {e}")
                preprocess_queue.put({})
        preprocess_thread = threading.Thread(target=preprocess_worker)
        preprocess_thread.start()

        # --- 【核心修改】定义三级流水线的队列 ---
        # 队列1：Whisper -> LLM
        transcription_queue: queue.Queue = queue.Queue(maxsize=os.cpu_count() or 1)
        # 队列2：LLM -> Matcher
        matching_queue: queue.Queue = queue.Queue(maxsize=os.cpu_count() or 1)
        
        remaining_sentences_map: Optional[Dict[str, str]] = None
        all_clips_results: List[Dict[str, Any]] = []
        lock = threading.Lock()
        
        # --- 2. 定义消费者工作函数 ---

        # 【新增】消费者1 / 生产者2: LLM 标准化工作线程
        def llm_normalization_worker():
            """消费来自 Whisper 的结果，执行 LLM 标准化，然后生产给匹配器。"""
            while True:
                task_item = transcription_queue.get()
                if task_item is None:
                    break
                
                video_path, all_words, clip_source_path = task_item
                try:
                    if all_words:
                        tasks[task_id]["message"] = f"[标准化中] {os.path.basename(video_path)} 的转写稿..."
                        # --- 【核心修改】调用 sm.normalize_words 封装函数 ---
                        norm_words = sm.normalize_words(all_words, method="llm")
                        matching_queue.put((video_path, all_words, norm_words, clip_source_path))
                    else:
                        # 如果没有识别结果，也放入一个空任务以保持流程
                        matching_queue.put((video_path, [], [], clip_source_path))
                except Exception as e:
                    print(f"LLM 标准化线程出错 ({os.path.basename(video_path)}): {e}")
                    # 放入空任务，避免阻塞
                    matching_queue.put((video_path, all_words, [], clip_source_path))
                finally:
                    transcription_queue.task_done()
        
        # 消费者2: 句子匹配工作线程
        def cpu_consumer_worker() -> None:
            """消费者线程: 并行执行CPU密集型任务 (句子匹配)。"""
            nonlocal remaining_sentences_map
            while True:
                task_item = matching_queue.get() # 从新的 matching_queue 获取
                if task_item is None:
                    break
                
                video_path, all_words, norm_words, clip_source_path = task_item
                original_filename = os.path.basename(video_path)
                
                if not norm_words: # 如果标准化失败，则跳过
                    matching_queue.task_done()
                    continue

                with lock:
                    if remaining_sentences_map is None:
                        tasks[task_id]["message"] = "语音转写进行中，等待目标句子标准化完成..."
                        norm_sentences_map = preprocess_queue.get()
                        remaining_sentences_map = norm_sentences_map.copy() if norm_sentences_map else {}

                    if not remaining_sentences_map:
                        matching_queue.task_done()
                        continue
                    current_sentences_to_find = remaining_sentences_map.copy()
                
                tasks[task_id]["message"] = f"[匹配中] {original_filename} (剩余 {len(current_sentences_to_find)} 句)"
                
                match_results = sm.find_best_match_for_all_sentences(
                    all_words=all_words,
                    norm_words=norm_words,
                    target_sentences_map=current_sentences_to_find,
                    confidence_threshold=50, 
                    search_mode='levenschtein'
                )

                if match_results:
                    with lock:
                        if remaining_sentences_map is None: continue 
                        for clip in match_results:
                            sentence = clip['sentence']
                            if sentence in remaining_sentences_map:
                                clip['clip_source_path'] = clip_source_path
                                clip['original_video_filename'] = original_filename
                                clip['video_url'] = f"/uploads/{original_filename}"
                                clip['note_id'] = sentence_to_note_id.get(sentence)
                                all_clips_results.append(clip)
                                del remaining_sentences_map[sentence]
                matching_queue.task_done()

        # --- 3. 启动消费者线程池 ---
        num_cpu_consumers = os.cpu_count() or 2
        # 【核心修改】为 LLM 和 CPU 分别创建线程池
        llm_threads = [threading.Thread(target=llm_normalization_worker, daemon=True) for _ in range(num_cpu_consumers)]
        matcher_threads = [threading.Thread(target=cpu_consumer_worker, daemon=True) for _ in range(num_cpu_consumers)]
        
        for t in llm_threads: t.start()
        for t in matcher_threads: t.start()

        # --- 4. 主线程作为生产者1，串行执行GPU任务 (Whisper) ---
        for i, video_path in enumerate(video_paths):
            with lock:
                if remaining_sentences_map is not None and not remaining_sentences_map:
                    tasks[task_id]["message"] = "所有句子已找到，提前结束转写。"
                    break
            
            original_filename = os.path.basename(video_path)
            
            audio_for_transcription = video_path
            if enable_separation:
                tasks[task_id]["message"] = f"[分离中] {original_filename}: 正在提取人声..."
                # --- 【关键修改】使用 Task ID 和文件名创建唯一的输出目录 ---
                video_base_name = os.path.splitext(original_filename)[0]
                sep_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'separated', task_id, video_base_name)
                
                # 调用 sentence_matching 中的分离函数
                vocals_path = sm.separate_vocals(video_path, output_dir=sep_output_dir)
                
                if vocals_path and os.path.exists(vocals_path):
                    audio_for_transcription = vocals_path
                    clip_source = vocals_path # 如果分离成功，后续裁剪将使用纯人声
                    print(f"音源分离成功: {vocals_path}")
                else:
                    print(f"警告: {original_filename} 音源分离失败，继续使用原音频。")
            
            tasks[task_id]["message"] = f"[转写 {i+1}/{len(video_paths)}] {original_filename}: 正在识别..."
            # --- 【核心修改】调用解耦后的函数，不执行标准化 ---
            transcription_result = sm.transcribe_and_normalize_audio(
                audio_path=audio_for_transcription,
                model_path=r"D:\ACGN\gal\whisper\models\faster-whisper-medium",
                device="cuda", compute_type="float16",
                perform_normalization=False
            )
            
            if transcription_result:
                all_words, _ = transcription_result # 第二个返回值为 None
                clip_source = video_path # 简化 clip_source 逻辑
                transcription_queue.put((video_path, all_words, clip_source))
            else:
                print(f"警告: 视频 {original_filename} 转写失败，已跳过。")
        
        # --- 5. 等待所有任务完成 (新的关闭顺序) ---
        transcription_queue.join()
        for _ in range(num_cpu_consumers): transcription_queue.put(None)
        for t in llm_threads: t.join()
        
        matching_queue.join()
        for _ in range(num_cpu_consumers): matching_queue.put(None)
        for t in matcher_threads: t.join()
        
        preprocess_thread.join()

        # --- 6. 整理并保存最终结果 ---
        if not all_clips_results:
            tasks[task_id] = {"status": "completed", "message": "处理完成，但未能从任何视频中匹配到任何句子。", "result_json_url": None}
            return

        json_filename = f"results_{task_id[:8]}.json"
        json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
        
        final_json_data = { "clips": sorted(all_clips_results, key=lambda x: x.get('score', 0), reverse=True) }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)

        tasks[task_id] = {
            "status": "completed",
            "message": f"处理完成！总共匹配到 {len(all_clips_results)} 个句子。",
            "result_json_url": f"/results/{json_filename}",
        }

    except Exception as e:
        traceback.print_exc()
        tasks[task_id] = {"status": "error", "error": f"后台处理时发生严重错误: {e}"}

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    host = "127.0.0.1"
    port = 5000
    print(f"--- 服务器正在启动 ---")
    print(f"请在浏览器中打开: http://{host}:{port}")
    serve(app, host=host, port=port)